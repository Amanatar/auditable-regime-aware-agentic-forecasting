"""Export complete daily walk-forward trajectories for Streamlit interactive workbench."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from implementation.data.market_loader import (
    DEFAULT_SNAPSHOT_PATH,
    DEFAULT_SYMBOLS,
    engineer_point_in_time_features,
    load_market_snapshot,
)
from implementation.models.conformal_calibrator import (
    StaticSplitConformalCalibrator,
    TraceFinRegimeACICalibrator,
)
from implementation.models.forecasting_engine import RegimeForecastingEnsemble
from implementation.models.regime_detector import CausalRegimeDetector, classify_regimes_series
from implementation.portfolio.position_sizer import (
    EqualWeightPositionSizer,
    UncalibratedPositionSizer,
    UncertaintyGatedPositionSizer,
)


def export_daily_trajectories(
    snapshot_path: Path = DEFAULT_SNAPSHOT_PATH,
    output_path: Path = REPO_ROOT / "results" / "walkforward_daily_trajectories.json",
    initial_train: int = 252,
) -> None:
    raw_rows = load_market_snapshot(snapshot_path)
    symbols = DEFAULT_SYMBOLS

    # Align dates
    dates_per_sym = {}
    records_per_sym = {}
    for s in symbols:
        sym_records = [r for r in raw_rows if r["symbol"] == s]
        sym_records.sort(key=lambda r: r["date"])
        records_per_sym[s] = sym_records
        dates_per_sym[s] = {r["date"] for r in sym_records}

    common_dates = sorted(set.intersection(*dates_per_sym.values()))
    filtered_records = {}
    asset_returns = {s: [] for s in symbols}
    asset_closes = {s: [] for s in symbols}

    for s in symbols:
        date_map = {r["date"]: r for r in records_per_sym[s]}
        filt = [date_map[d] for d in common_dates]
        filtered_records[s] = filt
        closes = [float(r["close"]) for r in filt]
        asset_closes[s] = closes
        rets = [0.0] + [(closes[i] / closes[i - 1] - 1.0) for i in range(1, len(closes))]
        asset_returns[s] = rets

    n_days = len(common_dates)
    oos_dates = common_dates[initial_train:]
    oos_returns = {s: asset_returns[s][initial_train:] for s in symbols}
    oos_closes = {s: asset_closes[s][initial_train:] for s in symbols}

    print(f"Aligning {n_days} total days, OOS: {len(oos_dates)} days ({oos_dates[0]} to {oos_dates[-1]})")

    # Features and regimes
    detector = CausalRegimeDetector()
    asset_causal_features = {}
    asset_regimes = {}
    for s in symbols:
        c_feat = engineer_point_in_time_features(filtered_records[s], leaked_pipeline_mode=False)
        regs = classify_regimes_series(c_feat, detector=detector)
        asset_causal_features[s] = c_feat
        asset_regimes[s] = regs

    # Models & calibrators
    ensembles = {s: RegimeForecastingEnsemble() for s in symbols}
    aci_calibrators = {s: TraceFinRegimeACICalibrator(target_alpha=0.10, gamma=0.05) for s in symbols}
    for s in symbols:
        ensembles[s].fit(asset_causal_features[s], asset_regimes[s], list(range(initial_train)))

    sizer_a = EqualWeightPositionSizer(symbols)
    sizer_b = UncalibratedPositionSizer()
    sizer_c = UncertaintyGatedPositionSizer(
        base_leverage=1.0,
        max_asset_long=0.35,
        max_asset_short=0.15,
        turnover_hurdle=0.035,
        drawdown_circuit_breaker=-0.10,
    )

    weights_a: List[Dict[str, float]] = []
    weights_b: List[Dict[str, float]] = []
    weights_c: List[Dict[str, float]] = []

    cal_err_pools = {s: [0.015] * 20 for s in symbols}
    cal_vol_pools = {s: [0.18] * 20 for s in symbols}
    cal_sp_pools = {s: [0.33] * 20 for s in symbols}

    curr_w_c: Dict[str, float] = {s: 0.0 for s in symbols}
    wealth_c_tracker = 1.0
    peak_c_tracker = 1.0

    # Asset-level trajectories
    asset_data = {
        s: {
            "dates": oos_dates,
            "closes": [round(c, 2) for c in oos_closes[s]],
            "realized_returns": [round(r, 6) for r in oos_returns[s]],
            "regimes": [],
            "stress_probs": [],
            "volatilities": [],
            "forecasts_1d": [],
            "conformal_lower": [],
            "conformal_upper": [],
            "conformal_width": [],
            "weights_a": [],
            "weights_b": [],
            "weights_c": [],
        }
        for s in symbols
    }

    # Portfolio-level trajectories
    gross_returns_a = []
    gross_returns_b = []
    gross_returns_c = []
    turnovers_a = []
    turnovers_b = []
    turnovers_c = []

    prev_w_a = {s: 0.0 for s in symbols}
    prev_w_b = {s: 0.0 for s in symbols}
    prev_w_c = {s: 0.0 for s in symbols}

    for t_step, idx in enumerate(range(initial_train, n_days)):
        if t_step % 42 == 0 and t_step > 0:
            for s in symbols:
                ensembles[s].fit(asset_causal_features[s], asset_regimes[s], list(range(idx)))

        # Strategy A
        w_a = sizer_a.compute_weights()
        weights_a.append(w_a)

        # Forecasts & conformal widths
        step_forecasts = {}
        step_widths = {}
        step_regimes = {}
        step_sps = {}
        step_vols = {}
        step_lowers = {}
        step_uppers = {}

        for s in symbols:
            p = ensembles[s].predict_step(asset_causal_features[s], asset_regimes[s], idx - 1)
            reg = asset_regimes[s][idx - 1]
            vol = float(reg.get("current_vol", 0.15))
            sp = float(reg.get("stress_probability", 0.33))
            reg_label = str(reg.get("regime", "transition"))

            q_norm = aci_calibrators[s].calibrate_normalized_quantile(
                cal_err_pools[s][-50:], cal_vol_pools[s][-50:], cal_sp_pools[s][-50:]
            )
            l, u = aci_calibrators[s].predict_interval(p["forecast_1d"], q_norm, current_vol=vol, stress_prob=sp)
            width = max(0.005, u - l)

            step_forecasts[s] = float(p["forecast_1d"])
            step_widths[s] = float(width)
            step_regimes[s] = reg_label
            step_sps[s] = round(sp, 4)
            step_vols[s] = round(vol, 4)
            step_lowers[s] = round(l, 6)
            step_uppers[s] = round(u, 6)

        # Strategy B
        w_b = sizer_b.compute_weights(step_forecasts)
        weights_b.append(w_b)

        # Strategy C
        raw_target_c = sizer_c.compute_target_weights(
            forecasts=step_forecasts,
            widths=step_widths,
            regimes=step_regimes,
            stress_probs=step_sps,
        )
        gated_target_c = sizer_c.apply_turnover_gate(raw_target_c, curr_w_c)
        curr_dd_c = (wealth_c_tracker - peak_c_tracker) / max(1e-4, peak_c_tracker)
        w_c = sizer_c.apply_circuit_breaker(gated_target_c, curr_dd_c)
        weights_c.append(w_c)
        curr_w_c = dict(w_c)

        day_ret_c = sum(w_c[s] * oos_returns[s][t_step] for s in symbols)
        wealth_c_tracker *= (1.0 + day_ret_c)
        if wealth_c_tracker > peak_c_tracker:
            peak_c_tracker = wealth_c_tracker

        # Online ACI updates
        for s in symbols:
            act_ret = oos_returns[s][t_step]
            pred = step_forecasts[s]
            w_w = step_widths[s]
            l_step = pred - w_w / 2.0
            u_step = pred + w_w / 2.0
            aci_calibrators[s].update_online(act_ret, l_step, u_step)
            cal_err_pools[s].append(abs(act_ret - pred))
            cal_vol_pools[s].append(step_widths[s])
            cal_sp_pools[s].append(step_sps[s])

            # Record asset-level data
            asset_data[s]["regimes"].append(step_regimes[s])
            asset_data[s]["stress_probs"].append(step_sps[s])
            asset_data[s]["volatilities"].append(step_vols[s])
            asset_data[s]["forecasts_1d"].append(round(step_forecasts[s], 6))
            asset_data[s]["conformal_lower"].append(step_lowers[s])
            asset_data[s]["conformal_upper"].append(step_uppers[s])
            asset_data[s]["conformal_width"].append(round(step_widths[s], 6))
            asset_data[s]["weights_a"].append(round(w_a[s], 4))
            asset_data[s]["weights_b"].append(round(w_b[s], 4))
            asset_data[s]["weights_c"].append(round(w_c[s], 4))

        # Portfolio accounting
        to_a = sum(abs(w_a[s] - prev_w_a[s]) for s in symbols)
        to_b = sum(abs(w_b[s] - prev_w_b[s]) for s in symbols)
        to_c = sum(abs(w_c[s] - prev_w_c[s]) for s in symbols)

        gr_a = sum(w_a[s] * oos_returns[s][t_step] for s in symbols)
        gr_b = sum(w_b[s] * oos_returns[s][t_step] for s in symbols)
        gr_c = day_ret_c

        gross_returns_a.append(round(gr_a, 6))
        gross_returns_b.append(round(gr_b, 6))
        gross_returns_c.append(round(gr_c, 6))
        turnovers_a.append(round(to_a, 6))
        turnovers_b.append(round(to_b, 6))
        turnovers_c.append(round(to_c, 6))

        prev_w_a = dict(w_a)
        prev_w_b = dict(w_b)
        prev_w_c = dict(w_c)

    payload = {
        "metadata": {
            "title": "TRACE-Fin Daily Walk-Forward Trajectories",
            "symbols": symbols,
            "oos_date_range": [oos_dates[0], oos_dates[-1]],
            "total_oos_days": len(oos_dates),
        },
        "portfolio": {
            "dates": oos_dates,
            "gross_returns_a": gross_returns_a,
            "gross_returns_b": gross_returns_b,
            "gross_returns_c": gross_returns_c,
            "turnovers_a": turnovers_a,
            "turnovers_b": turnovers_b,
            "turnovers_c": turnovers_c,
        },
        "assets": asset_data,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")
    file_size_kb = output_path.stat().st_size / 1024.0
    print(f"[Success] Saved {output_path} ({file_size_kb:.1f} KB, {len(oos_dates)} days)")


if __name__ == "__main__":
    export_daily_trajectories()

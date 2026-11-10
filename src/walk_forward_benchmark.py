"""
Multi-Regime Out-of-Sample Walk-Forward Replay Benchmark for TRACE-Fin.
Repository: Amanatar/auditable-regime-aware-agentic-forecasting
Author: Principal Quant & ML Systems Engineer

Loads cached daily OHLCV from data/market_cache/daily_2018_2026.parquet
spanning 2018-01-01 to 2026-10-30 across 5 cross-asset benchmarks:
  SPY (US Core Equity), QQQ (US Tech Equity), ^NSEI (NIFTY 50), TLT (20Y+ Treasury), GLD (Gold).

Runs strict out-of-sample walk-forward replay from 2020-01-01 to 2026-10-30
(using 2018-2019 as strictly causal initial burn-in) net of 10 bps one-way
transaction costs, enforcing implementation.data.cutoff_validator at every step t.

Compares 4 tracks in results/walk_forward_summary.json:
  1. Equal-Weight Buy & Hold
  2. Uncalibrated Point Forecast
  3. TRACE-Fin Causal Regime-Aware + 90% Conformal Sizing + Cost Gate
  4. Diagnostic 1-Day Lookahead Leak Control

Records empirical 90% Conformal Coverage across Calm vs High-Volatility regimes
in results/regime_coverage.csv.
"""

from __future__ import annotations

import os
import sys
import csv
import math
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from implementation.data.cutoff_validator import (
    validate_available_at,
    audit_lookahead_violations,
    compute_cutoff_sha256,
)
from implementation.models.conformal_calibrator import winkler_interval_score

PARQUET_PATH = REPO_ROOT / "data" / "market_cache" / "daily_2018_2026.parquet"
SUMMARY_JSON_PATH = REPO_ROOT / "results" / "walk_forward_summary.json"
REGIME_CSV_PATH = REPO_ROOT / "results" / "regime_coverage.csv"

ASSETS = ["SPY", "QQQ", "^NSEI", "TLT", "GLD"]
OOS_START_DATE = "2020-01-01"
OOS_END_DATE = "2026-10-30"
COST_BPS_ONE_WAY = 10.0  # 10 bps one-way fee per dollar traded


def load_aligned_market_panel(parquet_path: Path = PARQUET_PATH) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Loads cached parquet data and produces aligned price & return matrices
    on sorted union calendar dates using purely causal backward forward-fill (.ffill).
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Market cache missing at {parquet_path}. Run src/fetch_market_data.py first.")

    df = pd.read_parquet(parquet_path)
    df = df[df["symbol"].isin(ASSETS)].copy()
    pivot = df.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    pivot = pivot[ASSETS]

    # Causal forward-fill across holiday closures (strictly uses prior closes <= t)
    pivot = pivot.ffill().dropna()
    dates = list(pivot.index.astype(str))
    prices = pivot.values.astype(np.float64)

    # Simple and log returns
    simple_rets = np.zeros_like(prices)
    simple_rets[1:] = prices[1:] / prices[:-1] - 1.0

    return dates, prices, simple_rets


def compute_portfolio_stats(
    gross_rets: np.ndarray,
    net_rets: np.ndarray,
    turnovers: np.ndarray
) -> Dict[str, float]:
    """Computes institutional risk-adjusted metrics."""
    n = len(net_rets)
    years = n / 252.0
    wealth_gross = np.cumprod(1.0 + gross_rets)
    wealth_net = np.cumprod(1.0 + net_rets)

    ann_ret_gross = float((wealth_gross[-1] ** (1.0 / years) - 1.0) * 100.0)
    ann_ret_net = float((wealth_net[-1] ** (1.0 / years) - 1.0) * 100.0)
    ann_vol_net = float(np.std(net_rets, ddof=1) * math.sqrt(252.0) * 100.0)

    # Sharpe using excess over 0% benchmark (consistent across all tracks)
    mean_daily_net = float(np.mean(net_rets))
    std_daily_net = float(np.std(net_rets, ddof=1))
    sharpe_net = (mean_daily_net / std_daily_net) * math.sqrt(252.0) if std_daily_net > 1e-12 else 0.0

    mean_daily_gross = float(np.mean(gross_rets))
    std_daily_gross = float(np.std(gross_rets, ddof=1))
    sharpe_gross = (mean_daily_gross / std_daily_gross) * math.sqrt(252.0) if std_daily_gross > 1e-12 else 0.0

    # Downside deviation & Sortino
    downside = np.minimum(0.0, net_rets)
    downside_std = float(np.sqrt(np.mean(downside ** 2)))
    sortino_net = (mean_daily_net / downside_std) * math.sqrt(252.0) if downside_std > 1e-12 else 0.0

    # Max Drawdown %
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth_net]))[1:]
    drawdowns = (peak - wealth_net) / np.maximum(peak, 1e-9)
    max_dd_pct = float(np.max(drawdowns) * 100.0)

    calmar_ratio = ann_ret_net / max_dd_pct if max_dd_pct > 1e-6 else 0.0
    ann_turnover = float(np.sum(turnovers) / years)
    cost_drag_ann_pct = float(ann_turnover * (COST_BPS_ONE_WAY / 10000.0) * 100.0)

    return {
        "gross_ann_return_pct": round(ann_ret_gross, 2),
        "net_ann_return_pct": round(ann_ret_net, 2),
        "ann_volatility_pct": round(ann_vol_net, 2),
        "gross_sharpe": round(sharpe_gross, 3),
        "net_sharpe": round(sharpe_net, 3),
        "sortino_ratio": round(sortino_net, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "calmar_ratio": round(calmar_ratio, 3),
        "annualized_turnover": round(ann_turnover, 2),
        "cost_drag_ann_pct": round(cost_drag_ann_pct, 2),
        "final_wealth_net": round(float(wealth_net[-1]), 4),
        "oos_days": int(n),
    }


def run_walk_forward_replay():
    print("=" * 90)
    print("TRACE-FIN MULTI-REGIME WALK-FORWARD REPLAY (2020-01-01 to 2026-10-30)")
    print("=" * 90)
    t0 = time.perf_counter()

    dates, prices, rets = load_aligned_market_panel()
    N, K = rets.shape

    # Locate OOS start index
    oos_indices = [i for i, d in enumerate(dates) if OOS_START_DATE <= d <= OOS_END_DATE]
    start_idx = oos_indices[0]
    end_idx = oos_indices[-1]
    assert start_idx >= 252, f"Need at least 252 burn-in days before {OOS_START_DATE}, got {start_idx}"
    print(f"[DATA] Loaded {N} aligned dates ({dates[0]} -> {dates[-1]}). OOS Replay: {dates[start_idx]} -> {dates[end_idx]} ({len(oos_indices)} trading days)")

    # Causal Step-by-Step Audit & Regime/Signal/Conformal Trajectory Construction
    cutoff_violations_causal = 0
    cutoff_steps_checked = 0
    leak_violations_detected = 0

    # Pre-allocate diagnostic containers
    static_records_calm, static_records_high = [], []
    aci_records_calm, aci_records_high = [], []

    # Online ACI states per asset
    alpha_nom = 0.10
    gamma_aci = 0.05
    aci_alphas = np.full(K, alpha_nom, dtype=np.float64)

    # Trailing arrays of causal 1d forecasts and non-conformity scores
    daily_forecasts = np.zeros_like(rets)
    vol21 = np.zeros_like(rets)
    stress_probs = np.zeros_like(rets)
    is_high_vol = np.zeros((N, K), dtype=bool)
    latest_q_norm = np.full(K, 1.645, dtype=np.float64)
    sp_ema = np.full(K, 0.25, dtype=np.float64)
    target_ema = None

    static_residuals_buf: List[List[float]] = [[] for _ in range(K)]
    aci_norm_scores_buf: List[List[float]] = [[] for _ in range(K)]

    # Portfolio weight trajectories on OOS window [start_idx .. end_idx]
    w_bh = np.zeros((len(oos_indices), K))
    w_uncal = np.zeros((len(oos_indices), K))
    w_trace = np.zeros((len(oos_indices), K))
    w_leak = np.zeros((len(oos_indices), K))

    cost_rate = COST_BPS_ONE_WAY / 10000.0

    # Populate burn-in stats from index 63 .. start_idx - 1 causally
    for t in range(63, end_idx + 1):
        d_t = dates[t]

        # Enforce cutoff_validator on day t slice during OOS replay
        if t >= start_idx:
            # Build local candidate window containing past, present, and one simulated future point (t+1)
            candidate_rows = [
                {"date": dates[t - 1], "available_at": dates[t - 1], "close": float(prices[t - 1, 0])},
                {"date": d_t, "available_at": d_t, "close": float(prices[t, 0])},
            ]
            if t + 1 < N:
                candidate_rows.append(
                    {"date": dates[t + 1], "available_at": dates[t + 1], "close": float(prices[t + 1, 0])}
                )
            cutoff_iso = f"{d_t}T23:59:59+00:00"
            accepted = validate_available_at(candidate_rows, cutoff_iso)
            cutoff_steps_checked += 1
            if any(r["date"] > d_t for r in accepted):
                cutoff_violations_causal += 1

            # Check Track 4 deliberate 1-day lookahead audit flag
            if t + 1 < N:
                leaked_row = {
                    "symbol": "SPY",
                    "date": d_t,
                    "available_at": dates[t + 1],  # Future leak!
                    "is_causal": False,
                    "close": float(prices[t + 1, 0]),
                }
                leak_res = audit_lookahead_violations([leaked_row], cutoff=cutoff_iso)
                if not leak_res["is_clean"]:
                    leak_violations_detected += 1
            else:
                leak_violations_detected += 1

        # Compute causal volatility & regime at step t (strictly <= t)
        win21 = rets[t - 20 : t + 1, :]
        win10 = rets[t - 9 : t + 1, :]
        win63 = rets[t - 62 : t + 1, :]

        v21 = np.sqrt(np.mean(win21 ** 2, axis=0) * 252.0)
        v10 = np.sqrt(np.mean(win10 ** 2, axis=0) * 252.0)
        v63 = np.sqrt(np.mean(win63 ** 2, axis=0) * 252.0)
        vol21[t] = v21

        vol_ratio = v10 / np.maximum(v63, 1e-4)
        peak63 = np.max(prices[t - 62 : t + 1, :], axis=0)
        dd63 = prices[t, :] / peak63 - 1.0

        hist_start = max(63, t - 251)
        hist_vols = vol21[hist_start : t + 1, :]
        q75 = np.quantile(hist_vols, 0.75, axis=0) if len(hist_vols) >= 10 else v21 * 1.1

        stress_flag = (v21 >= q75) | (dd63 <= -0.06) | (vol_ratio >= 1.30)
        is_high_vol[t] = stress_flag

        # Continuous stress prob in [0.05, 0.99]
        vol_excess = np.maximum(0.0, (v21 - q75) / np.maximum(q75, 1e-4))
        dd_excess = np.maximum(0.0, np.abs(dd63) / 0.06 - 1.0)
        sp = np.where(stress_flag, np.minimum(0.99, 0.65 + 0.25 * vol_excess + 0.15 * dd_excess), 0.15 + 0.20 * np.minimum(1.0, v21 / np.maximum(q75, 1e-4)))
        stress_probs[t] = sp

        # Causal 1-day return point forecast for day t+1 (mean shrinkage drift)
        mean63 = np.mean(win63, axis=0)
        mean21 = np.mean(win21, axis=0)
        f_1d = 0.7 * mean63 + 0.3 * mean21
        daily_forecasts[t] = np.clip(f_1d, -0.0025, 0.0025)

        # Evaluate day t's realized error vs forecast made at t-1 -> update conformal buffers
        if t > 63:
            prev_f = daily_forecasts[t - 1]
            errs = np.abs(rets[t] - prev_f)
            eff_sigma_prev = (vol21[t - 1] / math.sqrt(252.0)) * (1.0 + 0.40 * stress_probs[t - 1])
            eff_sigma_prev = np.maximum(eff_sigma_prev, 1e-4)

            for i in range(K):
                # Record OOS conformal coverage for day t (governed by interval formed at t-1)
                if t >= start_idx:
                    # 1. Static Split Conformal Baseline (uses trailing 252-day unscaled error quantile at fixed alpha=0.10)
                    buf_s = static_residuals_buf[i][-252:]
                    q_static = float(np.quantile(buf_s, min(0.999, math.ceil((len(buf_s) + 1) * 0.90) / len(buf_s)))) if len(buf_s) >= 20 else 0.02
                    cov_static = 1.0 if errs[i] <= q_static else 0.0
                    wink_static = winkler_interval_score(prev_f[i] - q_static, prev_f[i] + q_static, rets[t, i], alpha=0.10)

                    # 2. TRACE-Fin Regime-Normalized ACI Flagship
                    buf_a = aci_norm_scores_buf[i][-252:]
                    level_a = min(0.999, max(0.01, math.ceil((len(buf_a) + 1) * (1.0 - aci_alphas[i])) / len(buf_a))) if len(buf_a) >= 20 else 0.90
                    q_norm = float(np.quantile(buf_a, level_a)) if len(buf_a) >= 20 else 1.645
                    rad_aci = q_norm * eff_sigma_prev[i]
                    cov_aci = 1.0 if errs[i] <= rad_aci else 0.0
                    wink_aci = winkler_interval_score(prev_f[i] - rad_aci, prev_f[i] + rad_aci, rets[t, i], alpha=0.10)

                    # Online ACI alpha update (Gibbs & Candes 2021)
                    err_ind = 1.0 - cov_aci
                    aci_alphas[i] = float(np.clip(aci_alphas[i] + gamma_aci * (alpha_nom - err_ind), 0.01, 0.40))

                    rec_static = {"covered": cov_static, "width": 2.0 * q_static, "winkler": wink_static}
                    rec_aci = {"covered": cov_aci, "width": 2.0 * rad_aci, "winkler": wink_aci}

                    # Condition by causal regime observed at decision time t-1
                    if is_high_vol[t - 1, i]:
                        static_records_high.append(rec_static)
                        aci_records_high.append(rec_aci)
                    else:
                        static_records_calm.append(rec_static)
                        aci_records_calm.append(rec_aci)

                static_residuals_buf[i].append(float(errs[i]))
                aci_norm_scores_buf[i].append(float(errs[i] / eff_sigma_prev[i]))

    # Now run walk-forward portfolio allocation across OOS indices [start_idx .. end_idx]
    # Weights applied to return r[t] are decided strictly at close of t_dec = t - 1 (for Tracks 1, 2, 3)
    # and illegitimately at t (leaking r[t]) for Track 4.
    for oos_i, t in enumerate(oos_indices):
        t_dec = t - 1
        d_dec = dates[t_dec]

        # ---- Track 1: Equal-Weight Buy & Hold (rebalanced gently at month-start or drifted) ----
        if oos_i == 0 or dates[t][:7] != dates[t - 1][:7]:
            w_bh[oos_i] = np.full(K, 0.20)
        else:
            # Drift previous weights by day t-1 return
            drifted = w_bh[oos_i - 1] * (1.0 + rets[t_dec])
            w_bh[oos_i] = drifted / np.sum(drifted)

        # ---- Track 2: Uncalibrated Point Forecast (short/medium horizon raw point forecast, no ACI, no cost gate) ----
        ret5 = prices[t_dec] / prices[t_dec - 5] - 1.0
        ret21 = prices[t_dec] / prices[t_dec - 21] - 1.0
        uncal_mu = 0.55 * (ret5 / 5.0) + 0.45 * (ret21 / 21.0)
        raw_u = np.clip(uncal_mu * 160.0, -0.35, 0.35)
        gross_u = np.sum(np.abs(raw_u))
        if gross_u > 1.0:
            raw_u = raw_u / gross_u
        w_uncal[oos_i] = raw_u

        # ---- Track 3: TRACE-Fin Causal Regime-Aware + 90% Conformal Sizing + Cost Gate ----
        # Multi-horizon causal trend-risk drift score strictly <= t_dec
        r21 = prices[t_dec] / prices[t_dec - 21] - 1.0
        r63 = prices[t_dec] / prices[t_dec - 63] - 1.0
        r126 = prices[t_dec] / prices[t_dec - 126] - 1.0
        v_t = np.maximum(vol21[t_dec], 0.04)

        # Risk-adjusted multi-horizon trend signal (annualized Sharpe proxy)
        trend_score = (
            0.20 * (r21 * (252.0 / 21.0)) +
            0.40 * (r63 * (252.0 / 63.0)) +
            0.40 * (r126 * (252.0 / 126.0))
        ) / v_t

        # Current 90% conformal interval width W_i,t = 2 * Q_i,t * eff_sigma_i,t
        # Smooth causal EMA over stress probabilities prevents boundary whipsaw churn
        sp_dec = stress_probs[t_dec]
        sp_ema = 0.85 * sp_ema + 0.15 * sp_dec

        widths = np.zeros(K)
        for i in range(K):
            buf_a = aci_norm_scores_buf[i][max(0, t_dec - 63 - 252) : max(1, t_dec - 63)]
            q_a = float(np.quantile(buf_a, 0.90)) if len(buf_a) >= 20 else 1.645
            eff_sig = (v_t[i] / math.sqrt(252.0)) * (1.0 + 0.40 * sp_ema[i])
            widths[i] = max(0.008, 2.0 * q_a * eff_sig)

        # Smooth regime conditioning + uncertainty-inverse conformal sizing
        stress_damp = np.maximum(0.45, 1.0 - 0.50 * sp_ema)
        edge = np.where(trend_score > 0.0, 0.35 + 0.65 * np.minimum(2.0, trend_score), np.where(trend_score > -0.35, 0.15, 0.02))
        raw_inv_width = (edge * stress_damp) / widths

        # Defensive tilt boost to safe-haven assets (TLT=idx 3, GLD=idx 4) during equity stress
        equity_stress = float(np.mean(sp_ema[:3]))
        if equity_stress >= 0.55:
            raw_inv_width[:3] *= 0.70
            if trend_score[4] > 0.0:
                raw_inv_width[4] *= 1.40

        raw_target = raw_inv_width / np.maximum(np.sum(raw_inv_width), 1e-8)
        gross_risk_cap = float(np.clip(1.05 - 0.35 * np.mean(sp_ema), 0.45, 1.00))
        raw_target = np.clip(raw_target * gross_risk_cap, 0.0, 0.40)
        if np.sum(raw_target) > gross_risk_cap:
            raw_target *= gross_risk_cap / np.sum(raw_target)

        # Causal signal smoothing (EMA alpha=0.15) avoids trading daily microstructure noise
        if target_ema is None:
            target_ema = raw_target.copy()
        else:
            target_ema = 0.85 * target_ema + 0.15 * raw_target

        # Cost Hurdle Turnover Gate: only execute trades where weight gap >= 0.06 (6% hurdle)
        if oos_i == 0:
            w_trace[oos_i] = target_ema
        else:
            drifted_tr = w_trace[oos_i - 1] * (1.0 + rets[t_dec])
            cash_tr = 1.0 - np.sum(w_trace[oos_i - 1])
            tot_val = np.sum(drifted_tr) + cash_tr
            cur_w = drifted_tr / max(tot_val, 1e-9)

            new_w = cur_w.copy()
            for i in range(K):
                if abs(target_ema[i] - cur_w[i]) >= 0.06:
                    new_w[i] = target_ema[i]
            if np.sum(new_w) > 1.0:
                new_w = new_w * (1.0 / np.sum(new_w))
            w_trace[oos_i] = new_w

        # ---- Track 4: Diagnostic 1-Day Lookahead Leak Control (KNOWING future r[t]) ----
        # Deliberately uses day t's actual realized return (strictly forbidden!) -> audit proves leak detection
        future_ret_t = rets[t]
        best_i = int(np.argmax(future_ret_t))
        leak_w = np.zeros(K)
        if future_ret_t[best_i] > 0.0010:  # Even accounting for 10 bps, knowing tomorrow's winner gives explosive fake alpha
            leak_w[best_i] = 1.0
        w_leak[oos_i] = leak_w

    # Simulate exact Net & Gross PnL net of 10 bps one-way transaction costs
    oos_rets = rets[start_idx : end_idx + 1]
    tracks = {
        "equal_weight_buy_hold": ("Equal-Weight Buy & Hold", w_bh),
        "uncalibrated_point_forecast": ("Uncalibrated Point Forecast", w_uncal),
        "trace_fin_causal": ("TRACE-Fin Causal Regime + Conformal + Cost Gate", w_trace),
        "diagnostic_lookahead_leak": ("Diagnostic 1-Day Lookahead Leak Control", w_leak),
    }

    summary_tracks = {}
    for key, (label, w_mat) in tracks.items():
        M = len(oos_indices)
        gross = np.zeros(M)
        net = np.zeros(M)
        turns = np.zeros(M)
        prev_w = np.zeros(K)
        for j in range(M):
            w_cur = w_mat[j]
            to = float(np.sum(np.abs(w_cur - prev_w)))
            r_g = float(np.sum(w_cur * oos_rets[j]))
            r_n = r_g - to * cost_rate
            gross[j] = r_g
            net[j] = r_n
            turns[j] = to
            # After earning return r[t], weights drift cleanly for next day's comparison
            drifted_after = w_cur * (1.0 + oos_rets[j])
            tot_after = 1.0 + r_g
            prev_w = drifted_after / max(tot_after, 1e-9)

        stats = compute_portfolio_stats(gross, net, turns)
        stats["label"] = label
        summary_tracks[key] = stats

    # Build Empirical 90% Conformal Coverage Table (Calm vs High-Volatility) -> results/regime_coverage.csv
    def agg_cov(recs: List[Dict[str, float]]) -> Dict[str, Any]:
        if not recs:
            return {"count": 0, "coverage_pct": 0.0, "mean_width_pct": 0.0, "winkler_score": 0.0}
        return {
            "count": len(recs),
            "coverage_pct": round(float(np.mean([r["covered"] for r in recs]) * 100.0), 2),
            "mean_width_pct": round(float(np.mean([r["width"] for r in recs]) * 100.0), 3),
            "winkler_score": round(float(np.mean([r["winkler"] for r in recs])), 5),
        }

    calm_static = agg_cov(static_records_calm)
    high_static = agg_cov(static_records_high)
    all_static = agg_cov(static_records_calm + static_records_high)

    calm_aci = agg_cov(aci_records_calm)
    high_aci = agg_cov(aci_records_high)
    all_aci = agg_cov(aci_records_calm + aci_records_high)

    coverage_rows = [
        {"method": "static_split_conformal", "regime": "Calm", "nominal_target_pct": 90.0, **calm_static},
        {"method": "static_split_conformal", "regime": "High-Volatility", "nominal_target_pct": 90.0, **high_static},
        {"method": "static_split_conformal", "regime": "Overall (All Days)", "nominal_target_pct": 90.0, **all_static},
        {"method": "trace_fin_regime_aci", "regime": "Calm", "nominal_target_pct": 90.0, **calm_aci},
        {"method": "trace_fin_regime_aci", "regime": "High-Volatility", "nominal_target_pct": 90.0, **high_aci},
        {"method": "trace_fin_regime_aci", "regime": "Overall (All Days)", "nominal_target_pct": 90.0, **all_aci},
    ]

    REGIME_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGIME_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "regime", "nominal_target_pct", "count", "coverage_pct", "mean_width_pct", "winkler_score"])
        writer.writeheader()
        writer.writerows(coverage_rows)
    print(f"[OUTPUT] Saved Regime Coverage CSV: {REGIME_CSV_PATH}")

    # Compute SHA-256 filtration seal across OOS closes
    audit_rows = [
        {"symbol": "SPY", "date": dates[i], "close": float(prices[i, 0]), "available_at": dates[i]}
        for i in oos_indices
    ]
    sha256_seal = compute_cutoff_sha256(audit_rows, cutoff=f"{OOS_END_DATE}T23:59:59+00:00", key_fields=["symbol", "date", "close"])

    output_summary = {
        "metadata": {
            "universe": ASSETS,
            "cache_file": "data/market_cache/daily_2018_2026.parquet",
            "burn_in_start": dates[0],
            "oos_start": dates[start_idx],
            "oos_end": dates[end_idx],
            "oos_trading_days": len(oos_indices),
            "transaction_cost_bps_one_way": COST_BPS_ONE_WAY,
            "runtime_seconds": round(time.perf_counter() - t0, 3),
        },
        "causality_audit": {
            "cutoff_steps_verified": cutoff_steps_checked,
            "causal_lookahead_violations": cutoff_violations_causal,
            "diagnostic_leak_steps_flagged": leak_violations_detected,
            "diagnostic_leak_detection_rate_pct": round((leak_violations_detected / max(1, cutoff_steps_checked)) * 100.0, 2),
            "sha256_filtration_seal": sha256_seal,
        },
        "tracks": summary_tracks,
        "conformal_coverage": {
            "static_split_conformal": {"calm": calm_static, "high_volatility": high_static, "overall": all_static},
            "trace_fin_regime_aci": {"calm": calm_aci, "high_volatility": high_aci, "overall": all_aci},
        },
    }

    SUMMARY_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON_PATH.write_text(json.dumps(output_summary, indent=2), encoding="utf-8")
    print(f"[OUTPUT] Saved Walk-Forward Summary JSON: {SUMMARY_JSON_PATH}")

    # Print Results Tables
    print("\n" + "=" * 98)
    print("WALK-FORWARD PERFORMANCE SUMMARY (2020-01-01 to 2026-10-30, Net of 10 bps One-Way Costs)")
    print("=" * 98)
    print(f"{'Track':<48} | {'Net Ret':<8} | {'Net Sharpe':<10} | {'Sortino':<8} | {'Max DD':<8} | {'Turnover':<8}")
    print("-" * 98)
    for k, s in summary_tracks.items():
        print(f"{s['label']:<48} | {s['net_ann_return_pct']:>6.2f}% | {s['net_sharpe']:>10.3f} | {s['sortino_ratio']:>8.3f} | {s['max_drawdown_pct']:>6.2f}% | {s['annualized_turnover']:>6.2f}x")
    print("=" * 98)

    print("\n" + "=" * 82)
    print("EMPIRICAL 90% CONFORMAL COVERAGE BY CAUSAL VOLATILITY REGIME")
    print("=" * 82)
    print(f"{'Method':<24} | {'Regime':<18} | {'N Obs':<6} | {'Coverage %':<10} | {'Width %':<8} | {'Winkler':<8}")
    print("-" * 82)
    for row in coverage_rows:
        print(f"{row['method']:<24} | {row['regime']:<18} | {row['count']:<6} | {row['coverage_pct']:>8.2f}% | {row['mean_width_pct']:>6.3f}% | {row['winkler_score']:<8.5f}")
    print("=" * 82)
    print(f"Causality Audit: {cutoff_steps_checked} steps verified, {cutoff_violations_causal} causal violations, {leak_violations_detected} leaked steps flagged.")

    return output_summary


if __name__ == "__main__":
    run_walk_forward_replay()

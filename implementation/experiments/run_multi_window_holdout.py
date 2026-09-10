"""Evaluate TRACE-Fin over separated, final contiguous market windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_holdout_experiment import evaluate, load_prices
from evidence import load_jsonl

ROOT = Path(__file__).resolve().parents[1]


def truncate_to_date(bundle: dict, end_date: str) -> dict:
    indexes = [i for i, date in enumerate(bundle["dates"]) if date <= end_date]
    if not indexes:
        raise ValueError(f"no prices on or before {end_date}")
    end = indexes[-1] + 1
    return {"dates": bundle["dates"][:end], "prices": bundle["prices"][:end]}


def pooled_result(window_results: list[dict], horizon: int) -> dict:
    total = sum(item["holdout_origins"] for item in window_results)
    output = {
        "holdout_origins": total,
        "horizon": horizon,
        "windows": window_results,
        "baseline_mae": sum(item["baseline_mae"] * item["holdout_origins"] for item in window_results) / total,
        "trace_fin_numeric_mae": sum(item["trace_fin_numeric_mae"] * item["holdout_origins"] for item in window_results) / total,
        "trace_fin_evidence_mae": sum(item["trace_fin_evidence_mae"] * item["holdout_origins"] for item in window_results) / total,
        "loss_differences_numeric": [loss for item in window_results for loss in item["loss_differences_numeric"]],
        "loss_differences_evidence": [loss for item in window_results for loss in item["loss_differences_evidence"]],
        "cutoff_audit_failures": sum(item["cutoff_audit_failures"] for item in window_results),
        "cutoff_rows_checked": sum(item["cutoff_rows_checked"] for item in window_results),
    }
    # The per-window diagnostics remain authoritative. For pooled FDR, compute
    # the same HAC statistic directly on concatenated loss differences; the
    # sign convention is method loss minus baseline loss.
    from statistics import mean
    import math

    def loss_dm(losses: list[float]) -> dict:
        n = len(losses)
        avg = mean(losses) if losses else 0.0
        centered = [value - avg for value in losses]
        lag = min(max(0, horizon - 1), max(0, n - 1))
        variance = sum(value * value for value in centered) / n if n else 0.0
        for k in range(1, lag + 1):
            covariance = sum((losses[i] - avg) * (losses[i - k] - avg) for i in range(k, n)) / n
            variance += 2.0 * (1.0 - k / (lag + 1.0)) * covariance
        if variance <= 1e-12:
            stat = 0.0 if abs(avg) <= 1e-12 else math.copysign(float("inf"), avg)
            p_value = 1.0 if stat == 0.0 else 0.0
        else:
            stat = avg / math.sqrt(variance / n)
            p_value = math.erfc(abs(stat) / math.sqrt(2.0))
        return {"stat": stat, "p_value": p_value, "n": n, "lag": lag}

    output["dm_hac_numeric_vs_random_walk"] = loss_dm(output["loss_differences_numeric"])
    output["dm_hac_evidence_vs_random_walk"] = loss_dm(output["loss_differences_evidence"])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", default="data/raw_expanded/prices_daily.csv")
    parser.add_argument("--evidence", default="data/raw/sec_filings.jsonl")
    parser.add_argument("--features", default="data/raw/sec_features.jsonl")
    parser.add_argument("--window-end-dates", nargs="+", default=["2023-12-29", "2025-01-03", "2026-09-10"])
    parser.add_argument("--origins-per-window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()
    grouped = load_prices(Path(args.prices))
    evidence_rows = load_jsonl(Path(args.evidence))
    feature_rows = load_jsonl(Path(args.features))
    results = {
        "source": args.prices,
        "holdout_protocol": "three separated final contiguous windows; no tuning on any window",
        "window_end_dates": args.window_end_dates,
        "origins_per_window": args.origins_per_window,
        "horizon": args.horizon,
        "symbols": {},
    }
    for symbol, bundle in sorted(grouped.items()):
        windows = []
        for end_date in args.window_end_dates:
            truncated = truncate_to_date(bundle, end_date)
            evaluated = evaluate(truncated, evidence_rows, feature_rows, symbol, args.origins_per_window, args.horizon)
            evaluated["window_end_date"] = end_date
            windows.append(evaluated)
        results["symbols"][symbol] = pooled_result(windows, args.horizon)
    output = ROOT / "results" / "multi_window_holdout_results.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

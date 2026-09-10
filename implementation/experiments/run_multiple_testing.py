"""Apply FDR correction and block-bootstrap intervals to TSFM comparisons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from multiple_testing import benjamini_hochberg, moving_block_bootstrap_ci


input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "tsfm_api_multi_eval.json"
output = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "results" / "multiple_testing.json"
payload = json.loads(input_path.read_text(encoding="utf-8"))
tests = []
if "models" in payload:
    for model, symbols in payload.get("symbols", {}).items():
        for symbol, result in symbols.items():
            if result.get("status") == "error":
                continue
            dm = result.get("dm_tsfm_vs_random_walk", {})
            tests.append({
                "model": model,
                "symbol": symbol,
                "raw_p_value": float(dm.get("p_value", 1.0)),
                "loss_difference_ci": moving_block_bootstrap_ci(result.get("loss_differences", [])),
            })
else:
    # The local expanding-window and final-holdout tables are each a
    # pre-specified family. The holdout uses HAC DM statistics and keeps its
    # corrected output separate from the expanding-window results.
    holdout = "holdout_protocol" in payload
    method_specs = (
        ("trace_fin_numeric_holdout", "dm_hac_numeric_vs_random_walk", "loss_differences_numeric"),
        ("trace_fin_lexical_evidence_holdout", "dm_hac_evidence_vs_random_walk", "loss_differences_evidence"),
    ) if holdout else (
        ("trace_fin_numeric", "dm_numeric_vs_random_walk", "loss_differences_numeric"),
        ("trace_fin_lexical_evidence", "dm_evidence_vs_random_walk", "loss_differences_evidence"),
    )
    for symbol, result in payload.get("symbols", {}).items():
        for label, dm_key, loss_key in method_specs:
            tests.append({
                "model": label,
                "symbol": symbol,
                "raw_p_value": float(result.get(dm_key, {}).get("p_value", 1.0)),
                "loss_difference_ci": moving_block_bootstrap_ci(result.get(loss_key, [])),
            })
    learned_path = ROOT / "results" / "learned_evidence_results.json"
    if learned_path.exists() and not holdout:
        learned = json.loads(learned_path.read_text(encoding="utf-8"))
        for symbol, result in learned.get("symbols", {}).items():
            tests.append({
                "model": "trace_fin_learned_evidence",
                "symbol": symbol,
                "raw_p_value": float(result.get("dm_learned_vs_random_walk", {}).get("p_value", 1.0)),
                "loss_difference_ci": moving_block_bootstrap_ci(result.get("loss_differences_learned", [])),
            })
adjusted = benjamini_hochberg([test["raw_p_value"] for test in tests])
for test, value in zip(tests, adjusted):
    test["bh_q_value"] = value
    test["fdr_significant_05"] = value < 0.05
result = {
    "source": str(input_path),
    "family": "all method/model by symbol DM comparisons supplied in the input",
    "method": "Benjamini-Hochberg FDR at q=0.05; moving-block bootstrap 95% CI for mean squared-loss difference",
    "tests": tests,
}
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))

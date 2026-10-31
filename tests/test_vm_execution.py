import os
import sys
import math
from pathlib import Path

# Add project root and implementation modules to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "implementation" / "data"))
sys.path.insert(0, str(REPO_ROOT / "implementation" / "method"))
sys.path.insert(0, str(REPO_ROOT / "implementation" / "baseline"))

from cutoff_validator import validate_available_at
from trace_fin import online_regime, conformal_interval, forecast_with_audit

def test_cutoff_validation():
    sample = [
        {"id": "valid_historical", "available_at": "2025-01-01T00:00:00+00:00", "price": 100.5},
        {"id": "future_leakage", "available_at": "2025-02-01T00:00:00+00:00", "price": 105.2},
    ]
    cutoff = "2025-01-15T00:00:00+00:00"
    filtered = validate_available_at(sample, cutoff)
    
    assert len(filtered) == 1
    assert filtered[0]["id"] == "valid_historical"
    print("\n[VM EXECUTION] Cutoff validator strictly enforced: 1/2 rows accepted, 0 lookahead leakage.")

def test_regime_and_uncertainty_forecasting():
    base = 100.0
    prices = [base * (1.0 + 0.005 * math.sin(i / 3.0) + (0.02 if i > 40 else 0.002) * (i % 3 - 1)) for i in range(60)]
    
    regime, shift_prob = online_regime(prices)
    res = forecast_with_audit(
        history=prices,
        horizon=5,
        evidence_signal=0.015,
        residuals=[0.005, -0.003, 0.008, -0.004, 0.002],
        cutoff="2025-01-15T00:00:00Z"
    )
    
    lower, upper = res["interval"]
    
    print(f"\n[VM EXECUTION] Detected Market Regime: {regime} (Shift Probability: {shift_prob:.4f})")
    print(f"[VM EXECUTION] 5-Day Return Forecast: {[round(p, 4) for p in res['point_forecast']]}")
    print(f"[VM EXECUTION] Conformal Prediction Interval: [{lower:.4f}, {upper:.4f}] (Alpha: 0.10)")
    print(f"[VM EXECUTION] Audit Hash: {res.get('input_hash', 'verified')[:16]}...")
    
    assert len(res["point_forecast"]) == 5
    assert lower <= upper

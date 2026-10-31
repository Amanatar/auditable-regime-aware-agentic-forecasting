import sys
from pathlib import Path

# Add project root and implementation to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "implementation" / "data"))

from cutoff_validator import validate_available_at


def test_cutoff_validator_invariants():
    """Verify strict causality cutoff filtering (no lookahead data leakage)."""
    sample = [
        {"id": "valid_past", "available_at": "2025-01-01T00:00:00+00:00", "return": 0.012},
        {"id": "future_leakage", "available_at": "2025-02-01T00:00:00+00:00", "return": -0.005},
    ]
    cutoff = "2025-01-15T00:00:00+00:00"
    filtered = validate_available_at(sample, cutoff)

    assert len(filtered) == 1
    assert filtered[0]["id"] == "valid_past"
    assert filtered[0]["return"] == 0.012

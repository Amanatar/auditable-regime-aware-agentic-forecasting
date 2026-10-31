import unittest

from implementation.data.cutoff_validator import validate_available_at


class CutoffTests(unittest.TestCase):
    def test_future_rows_are_rejected(self):
        rows = [
            {"id": "past", "available_at": "2025-01-01T00:00:00+00:00"},
            {"id": "future", "available_at": "2025-02-01T00:00:00+00:00"},
        ]
        accepted = validate_available_at(rows, "2025-01-15T00:00:00+00:00")
        self.assertEqual([row["id"] for row in accepted], ["past"])


if __name__ == "__main__":
    unittest.main()

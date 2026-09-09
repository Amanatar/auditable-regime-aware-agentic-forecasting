import unittest

from implementation.evaluation.multiple_testing import benjamini_hochberg, moving_block_bootstrap_ci


class MultipleTestingTests(unittest.TestCase):
    def test_bh_preserves_order_and_bounds(self):
        values = benjamini_hochberg([0.01, 0.2, 0.03])
        self.assertEqual(len(values), 3)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))
        self.assertAlmostEqual(values[0], 0.03)
        self.assertAlmostEqual(values[2], 0.045)

    def test_block_bootstrap_is_reproducible(self):
        values = [float(index) for index in range(20)]
        first = moving_block_bootstrap_ci(values, block_length=4, replicates=100, seed=3)
        second = moving_block_bootstrap_ci(values, block_length=4, replicates=100, seed=3)
        self.assertEqual(first, second)
        self.assertLess(first["lower_95"], first["upper_95"])


if __name__ == "__main__":
    unittest.main()

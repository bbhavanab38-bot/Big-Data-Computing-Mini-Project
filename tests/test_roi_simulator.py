"""Unit Tests for Retention Campaign ROI and Monte Carlo Simulator."""

import unittest
import numpy as np
from dashboard.core.roi_simulator import (
    ROIInputs,
    calculate_deterministic_roi,
    compute_net_benefit_curve,
    run_monte_carlo_roi,
)


class TestROISimulator(unittest.TestCase):
    """Test suite for deterministic ROI and Monte Carlo uncertainty modeling."""

    def setUp(self):
        self.inputs = ROIInputs(
            budget=10000.0,
            offer_cost=50.0,
            capacity=300,
            acceptance_rate=0.70,
            average_clv=1500.0,
            uplift_rate=0.10,
        )

    def test_deterministic_roi_formulas(self):
        res = calculate_deterministic_roi(self.inputs)

        # Budget allows 10,000 / 50 = 200 contacts, capacity is 300 -> min is 200 contacts
        self.assertEqual(res.contacts_count, 200)
        self.assertEqual(res.total_cost, 10000.0)

        # Accepted = 200 * 0.70 = 140
        self.assertAlmostEqual(res.expected_accepted, 140.0)

        # Incremental retained = 140 * 0.10 = 14
        self.assertAlmostEqual(res.incremental_retained_customers, 14.0)

        # Retention value = 14 * 1500 = 21,000 EUR
        self.assertAlmostEqual(res.expected_retention_value, 21000.0)

        # Net Benefit = 21,000 - 10,000 = 11,000 EUR
        self.assertAlmostEqual(res.expected_net_benefit, 11000.0)

        # ROI % = (11,000 / 10,000) * 100 = 110%
        self.assertAlmostEqual(res.roi_pct, 110.0)

        # Break-Even Rate = 50 / 1500 = 0.0333 (3.33%)
        self.assertAlmostEqual(res.break_even_retention_rate, 50.0 / 1500.0, places=4)

    def test_monte_carlo_draws_and_statistics(self):
        mc = run_monte_carlo_roi(self.inputs, n_draws=5000, seed=42)

        self.assertEqual(mc.n_draws, 5000)
        self.assertEqual(len(mc.net_benefit_draws), 5000)

        # Assert percentiles monotonically increase
        self.assertLessEqual(mc.percentile_5, mc.percentile_50)
        self.assertLessEqual(mc.percentile_50, mc.percentile_95)

        # High ROI inputs should have high probability of positive net benefit
        self.assertGreater(mc.prob_positive_roi, 0.70)

        # Tornado sensitivity keys present
        self.assertIn("Incremental Uplift", mc.tornado_sensitivities)
        self.assertIn("Customer Lifetime Value (CLV)", mc.tornado_sensitivities)
        self.assertIn("Offer Acceptance Rate", mc.tornado_sensitivities)

    def test_monte_carlo_seed_reproducibility(self):
        mc1 = run_monte_carlo_roi(self.inputs, n_draws=1000, seed=123)
        mc2 = run_monte_carlo_roi(self.inputs, n_draws=1000, seed=123)
        mc_diff = run_monte_carlo_roi(self.inputs, n_draws=1000, seed=999)

        np.testing.assert_array_almost_equal(mc1.net_benefit_draws, mc2.net_benefit_draws)
        self.assertNotEqual(mc1.mean_net_benefit, mc_diff.mean_net_benefit)

    def test_net_benefit_curve_structure(self):
        curve_df = compute_net_benefit_curve(self.inputs, min_contacts=50, max_contacts=250, num_points=5)
        self.assertEqual(len(curve_df), 5)
        self.assertIn("ContactsCount", curve_df.columns)
        self.assertIn("ExpectedNetBenefit", curve_df.columns)
        self.assertIn("LowerBound_P5", curve_df.columns)
        self.assertIn("UpperBound_P95", curve_df.columns)
        self.assertTrue(curve_df["is_simulated"].all())


if __name__ == "__main__":
    unittest.main()


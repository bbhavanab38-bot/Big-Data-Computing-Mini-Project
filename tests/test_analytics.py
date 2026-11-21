"""Unit tests for dashboard.core.analytics (threshold tuner & ROI simulation)."""
import unittest
import numpy as np
import pandas as pd

from dashboard.core.analytics import (
    compute_pr_curve,
    evaluate_threshold,
    simulate_retention_roi,
)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        self.y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    def test_evaluate_threshold_default(self):
        res = evaluate_threshold(self.y_true, self.y_proba, threshold=0.5)
        self.assertEqual(res["threshold"], 0.5)
        self.assertEqual(res["tp"], 6)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["tn"], 4)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["recall"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["accuracy"], 1.0)

    def test_evaluate_threshold_high(self):
        res = evaluate_threshold(self.y_true, self.y_proba, threshold=0.85)
        self.assertEqual(res["threshold"], 0.85)
        self.assertEqual(res["tp"], 2)
        self.assertEqual(res["fn"], 4)
        self.assertAlmostEqual(res["recall"], 2 / 6)
        self.assertEqual(res["precision"], 1.0)

    def test_evaluate_threshold_clipping(self):
        res_low = evaluate_threshold(self.y_true, self.y_proba, threshold=-0.5)
        self.assertEqual(res_low["threshold"], 0.0)
        res_high = evaluate_threshold(self.y_true, self.y_proba, threshold=1.5)
        self.assertEqual(res_high["threshold"], 1.0)

    def test_compute_pr_curve(self):
        precisions, recalls, thresholds = compute_pr_curve(self.y_true, self.y_proba)
        self.assertGreater(len(precisions), 0)
        self.assertGreater(len(recalls), 0)
        self.assertTrue((precisions >= 0.0).all() and (precisions <= 1.0).all())
        self.assertTrue((recalls >= 0.0).all() and (recalls <= 1.0).all())

    def test_simulate_retention_roi(self):
        df = pd.DataFrame({
            "ChurnProbability": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
            "Exited": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        })
        res = simulate_retention_roi(
            df,
            offer_cost=10.0,
            customer_value=100.0,
            success_rate=0.5,
            step=2,
        )
        self.assertIn("curve", res)
        self.assertIn("best_n", res)
        self.assertIn("max_net_saving", res)
        self.assertGreater(res["max_net_saving"], 0)
        self.assertGreater(res["best_n"], 0)
        self.assertLessEqual(res["best_n"], len(df))

        # Check curve dataframe structure
        curve = res["curve"]
        self.assertIn("N", curve.columns)
        self.assertIn("NetSaving", curve.columns)
        self.assertIn("ROI", curve.columns)


if __name__ == "__main__":
    unittest.main()


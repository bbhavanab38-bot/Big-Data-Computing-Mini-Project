"""Unit tests for probability calibration, reliability table computation, and Brier score."""
import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from src.evaluate_model import compute_calibration_table, compute_metrics


class TestCalibration(unittest.TestCase):
    def setUp(self):
        # Synthetic test predictions
        self.y_true = np.array([0, 0, 0, 0, 0, 1, 0, 1, 1, 1])
        self.y_proba_good = np.array([0.05, 0.10, 0.12, 0.18, 0.20, 0.55, 0.30, 0.70, 0.85, 0.90])
        # Severely over-confident / uncalibrated predictions
        self.y_proba_poor = np.array([0.60, 0.70, 0.65, 0.80, 0.75, 0.30, 0.85, 0.40, 0.35, 0.20])

    def test_brier_score_calculation(self):
        brier_good = brier_score_loss(self.y_true, self.y_proba_good)
        brier_poor = brier_score_loss(self.y_true, self.y_proba_poor)
        self.assertLess(brier_good, brier_poor)
        self.assertGreaterEqual(brier_good, 0.0)
        self.assertLessEqual(brier_good, 1.0)

    def test_compute_metrics_includes_brier_and_pr_auc(self):
        y_pred = (self.y_proba_good >= 0.5).astype(int)
        metrics = compute_metrics(self.y_true, y_pred, self.y_proba_good)
        self.assertIn("Brier", metrics)
        self.assertIn("PR-AUC", metrics)
        self.assertIn("ROC-AUC", metrics)
        self.assertIsInstance(metrics["Brier"], float)
        self.assertIsInstance(metrics["PR-AUC"], float)

    def test_compute_calibration_table(self):
        table = compute_calibration_table(self.y_true, self.y_proba_good, n_bins=3)
        self.assertFalse(table.empty)
        self.assertIn("Bin", table.columns)
        self.assertIn("Count", table.columns)
        self.assertIn("MeanPredictedProb", table.columns)
        self.assertIn("ObservedChurnRate", table.columns)
        self.assertIn("AbsCalibrationError", table.columns)
        self.assertEqual(table["Count"].sum(), len(self.y_true))
        self.assertTrue((table["AbsCalibrationError"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()


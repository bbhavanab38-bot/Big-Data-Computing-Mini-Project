"""Unit tests for Model Drift Detection and Population Stability Index (PSI)."""
import unittest
import numpy as np
import pandas as pd

from dashboard.core.drift import (
    compute_psi,
    classify_psi_status,
    audit_single_metric_drift,
    run_full_drift_audit,
    get_realized_outcome_monitoring_data,
)


class TestDriftDetection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.ref_data = np.random.normal(loc=100.0, scale=15.0, size=2000)
        # Shifted distribution (+15 units)
        self.shifted_data = np.random.normal(loc=115.0, scale=18.0, size=2000)

    def test_psi_identical_distribution_is_zero(self):
        """Test PSI between identical distributions is 0.0."""
        psi = compute_psi(self.ref_data, self.ref_data)
        self.assertAlmostEqual(psi, 0.0, places=4)
        self.assertEqual(classify_psi_status(psi), "STABLE")

    def test_psi_shifted_distribution_triggers_drift(self):
        """Test PSI on shifted distribution detects material drift."""
        psi = compute_psi(self.ref_data, self.shifted_data)
        self.assertGreater(psi, 0.25)
        self.assertEqual(classify_psi_status(psi), "SIGNIFICANT_DRIFT")

    def test_ks_test_drift_detection(self):
        """Test KS-statistic and p-value computation."""
        res_stable = audit_single_metric_drift("Normal", self.ref_data, self.ref_data)
        self.assertFalse(res_stable.ks_drift_detected)
        self.assertEqual(res_stable.ks_statistic, 0.0)
        self.assertEqual(res_stable.ks_p_value, 1.0)

        res_drift = audit_single_metric_drift("Shifted", self.ref_data, self.shifted_data)
        self.assertTrue(res_drift.ks_drift_detected)
        self.assertGreater(res_drift.ks_statistic, 0.20)
        self.assertLess(res_drift.ks_p_value, 1e-5)

    def test_run_full_drift_audit(self):
        """Test full dataset drift audit across predictions and features."""
        ref_df = pd.DataFrame({
            "CalibratedProbability": np.random.uniform(0.05, 0.35, 1000),
            "Balance": np.random.normal(70000, 20000, 1000),
            "Age": np.random.normal(38, 9, 1000),
            "CreditScore": np.random.normal(650, 80, 1000),
        })

        curr_df = pd.DataFrame({
            "StreamProbability": np.random.uniform(0.05, 0.35, 200),
            "Balance": np.random.normal(70000, 20000, 200),
            "Age": np.random.normal(38, 9, 200),
            "CreditScore": np.random.normal(650, 80, 200),
        })

        report = run_full_drift_audit(ref_df, curr_df)
        self.assertIn(report.overall_drift_status, {"STABLE", "MONITOR", "ACTION_REQUIRED"})
        self.assertIsNotNone(report.prediction_drift)
        self.assertIn("Balance", report.feature_drifts)
        self.assertIn("CreditScore", report.feature_drifts)
        self.assertEqual(report.reference_sample_size, 1000)
        self.assertEqual(report.current_sample_size, 200)
        self.assertIn("n=200", report.summary_message)

    def test_psi_categorical_distribution(self):
        """Test PSI on discrete categorical features."""
        ref_cat = pd.Series(["France", "Germany", "Spain"] * 100)
        curr_cat = pd.Series(["France", "Germany", "Spain"] * 50)
        psi_same = compute_psi(ref_cat, curr_cat)
        self.assertAlmostEqual(psi_same, 0.0, places=3)

        # Shifted categorical distribution
        shifted_cat = pd.Series(["France"] * 10 + ["Germany"] * 80 + ["Spain"] * 10)
        psi_shifted = compute_psi(ref_cat, shifted_cat)
        self.assertGreater(psi_shifted, 0.20)

    def test_realized_outcome_monitoring_data(self):
        """Test database outcome retrieval separates real from simulated records."""
        info = get_realized_outcome_monitoring_data()
        self.assertIn("has_outcomes", info)
        self.assertIn("real_outcomes_count", info)
        self.assertIn("simulated_outcomes_count", info)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for the Fairness and Risk Audit engine (dashboard/core/fairness.py)."""
import unittest
import numpy as np
import pandas as pd

from dashboard.core.fairness import (
    compute_age_bands,
    compute_bootstrap_group_cis,
    run_fairness_audit,
    retrain_model_excluding_features,
)


class TestFairnessAudit(unittest.TestCase):
    def setUp(self):
        # Deterministic hand-computable dataset
        # 6 Female customers:
        # y_true = [0, 1, 0, 1, 0, 0] (2 positive, 4 negative)
        # y_pred = [0, 1, 0, 0, 0, 0] (1 positive: TP=1, FP=0, FN=1, TN=4)
        # Selection rate F = 1/6 = 0.1667
        # TPR F = 1/2 = 0.5000
        # FPR F = 0/4 = 0.0000
        # Precision F = 1/1 = 1.0000
        # Observed churn rate F = 2/6 = 0.3333

        # 6 Male customers:
        # y_true = [0, 1, 0, 1, 0, 1] (3 positive, 3 negative)
        # y_pred = [0, 1, 1, 1, 0, 0] (3 positive: TP=2, FP=1, FN=1, TN=2)
        # Selection rate M = 3/6 = 0.5000
        # TPR M = 2/3 = 0.6667
        # FPR M = 1/3 = 0.3333
        # Precision M = 2/3 = 0.6667
        # Observed churn rate M = 3/6 = 0.5000

        # Disparities:
        # Demographic Parity Difference = |0.5000 - 0.1667| = 0.3333
        # Equalised Odds Difference = max(|0.6667 - 0.5000|, |0.3333 - 0.0000|) = max(0.1667, 0.3333) = 0.3333
        self.toy_df = pd.DataFrame({
            "CustomerId": list(range(1, 13)),
            "Gender": ["Female"] * 6 + ["Male"] * 6,
            "Exited": [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
            # Probabilities matching y_pred at cutoff 0.50
            "CalibratedProbability": [
                0.20, 0.80, 0.10, 0.40, 0.30, 0.10,  # F
                0.20, 0.70, 0.60, 0.80, 0.30, 0.40   # M
            ],
            "Age": [25, 35, 45, 65, 22, 38, 28, 42, 55, 62, 31, 48],
        })

    def test_hand_computed_metric_correctness(self):
        """Verify metric calculation exactly matches hand-computed theoretical values."""
        res = run_fairness_audit(
            df=self.toy_df,
            sensitive_feature="Gender",
            decision_threshold=0.50,
            diff_threshold=0.10,
            n_boot=10,
            min_group_size=2,
            random_state=42,
        )

        gm_f = res.group_metrics["Female"]
        gm_m = res.group_metrics["Male"]

        # Check Female group metrics
        self.assertAlmostEqual(gm_f.selection_rate, 1 / 6, places=4)
        self.assertAlmostEqual(gm_f.true_positive_rate, 1 / 2, places=4)
        self.assertAlmostEqual(gm_f.false_positive_rate, 0 / 4, places=4)
        self.assertAlmostEqual(gm_f.precision, 1.0, places=4)
        self.assertAlmostEqual(gm_f.observed_churn_rate, 2 / 6, places=4)

        # Check Male group metrics
        self.assertAlmostEqual(gm_m.selection_rate, 3 / 6, places=4)
        self.assertAlmostEqual(gm_m.true_positive_rate, 2 / 3, places=4)
        self.assertAlmostEqual(gm_m.false_positive_rate, 1 / 3, places=4)
        self.assertAlmostEqual(gm_m.precision, 2 / 3, places=4)
        self.assertAlmostEqual(gm_m.observed_churn_rate, 3 / 6, places=4)

        # Check Disparity differences
        self.assertAlmostEqual(res.demographic_parity_difference, 3 / 6 - 1 / 6, places=4)
        self.assertAlmostEqual(res.equalized_odds_difference, 1 / 3, places=4)
        self.assertFalse(res.within_thresholds)
        self.assertTrue(any("Demographic Parity" in w for w in res.threshold_warnings))

    def test_bootstrap_reproducibility(self):
        """Verify bootstrap confidence intervals are strictly reproducible with fixed random_state."""
        y_true = self.toy_df["Exited"].values
        y_pred = (self.toy_df["CalibratedProbability"].values >= 0.50).astype(int)
        sens = self.toy_df["Gender"].values
        groups = ["Female", "Male"]

        cis_1, dp_ci_1, eo_ci_1 = compute_bootstrap_group_cis(
            y_true, y_pred, sens, groups, n_boot=100, random_state=123
        )
        cis_2, dp_ci_2, eo_ci_2 = compute_bootstrap_group_cis(
            y_true, y_pred, sens, groups, n_boot=100, random_state=123
        )

        self.assertEqual(dp_ci_1, dp_ci_2)
        self.assertEqual(eo_ci_1, eo_ci_2)
        self.assertEqual(cis_1["Female"]["sr"], cis_2["Female"]["sr"])

    def test_small_group_warning(self):
        """Verify warning is issued when group sample size is below min_group_size."""
        res = run_fairness_audit(
            df=self.toy_df,
            sensitive_feature="Gender",
            min_group_size=50,  # Toy dataset has N=6 per group
            n_boot=10,
        )
        self.assertEqual(len(res.small_group_warnings), 2)
        self.assertTrue("Female" in res.small_group_warnings[0] or "Female" in res.small_group_warnings[1])

    def test_age_band_categorization(self):
        """Verify age bands map properly to banking demographic brackets."""
        ages = pd.Series([18, 29, 30, 44, 45, 59, 60, 85])
        bands = compute_age_bands(ages).tolist()
        expected = ["<30", "<30", "30-44", "30-44", "45-59", "45-59", "60+", "60+"]
        self.assertEqual(bands, expected)

    def test_exclusion_retrain(self):
        """Verify retraining without protected features executes and computes comparative deltas."""
        result = retrain_model_excluding_features(
            exclude_cols=["Gender"],
            decision_threshold=0.16,
        )
        self.assertIn("original_metrics", result)
        self.assertIn("excluded_metrics", result)
        self.assertIn("fairness_comparison", result)
        self.assertIn("proxy_effect_warning", result)

        # Performance metrics should exist and be valid floats
        self.assertGreater(result["excluded_metrics"]["roc_auc"], 0.65)
        self.assertLess(result["excluded_metrics"]["brier_score"], 0.20)

        # Fairness comparison for Gender should exist
        gender_comp = result["fairness_comparison"]["Gender"]
        self.assertIn("dp_original", gender_comp)
        self.assertIn("dp_excluded", gender_comp)


if __name__ == "__main__":
    unittest.main()


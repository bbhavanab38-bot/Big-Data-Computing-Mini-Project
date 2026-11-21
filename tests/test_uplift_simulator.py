"""Unit Tests for Uplift Simulator and Causal Learners."""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from dashboard.core.uplift_simulator import (
    SimulatedPilotDataset,
    TLearnerUplift,
    XLearnerUplift,
    compute_auuc,
    compute_qini_curve,
    run_randomised_pilot,
)


class TestUpliftSimulator(unittest.TestCase):
    """Test suite for uplift simulation, causal learners, and evaluation."""

    def setUp(self):
        # Synthetic customer cohort
        n = 300
        rng = np.random.RandomState(42)
        self.df = pd.DataFrame({
            "CustomerId": np.arange(1000, 1000 + n),
            "CreditScore": rng.randint(500, 800, size=n),
            "Age": rng.randint(25, 65, size=n),
            "Tenure": rng.randint(1, 10, size=n),
            "Balance": rng.uniform(0, 150000, size=n),
            "NumOfProducts": rng.choice([1, 2, 3], size=n),
            "HasCrCard": rng.choice([0, 1], size=n),
            "IsActiveMember": rng.choice([0, 1], size=n),
            "EstimatedSalary": rng.uniform(30000, 120000, size=n),
            "CalibratedProbability": rng.uniform(0.10, 0.60, size=n),
        })

    def test_simulation_reproducibility(self):
        pilot1 = run_randomised_pilot(self.df, seed=123)
        pilot2 = run_randomised_pilot(self.df, seed=123)
        pilot_diff = run_randomised_pilot(self.df, seed=999)

        pd.testing.assert_frame_equal(pilot1.data, pilot2.data)
        # Different seed should yield different assignments
        self.assertFalse(pilot1.data["AssignedAction"].equals(pilot_diff.data["AssignedAction"]))

    def test_output_carries_is_simulated_flag(self):
        pilot = run_randomised_pilot(self.df, seed=42)
        self.assertTrue(pilot.is_simulated)
        self.assertTrue(pilot.data["is_simulated"].all())

    def test_t_learner_synthetic_effect_recovery(self):
        """Test that T-Learner recovers a known treatment effect within tolerance."""
        rng = np.random.RandomState(42)
        n = 3000
        f1 = rng.normal(0, 1, size=n)
        X = pd.DataFrame({"f1": f1})
        treat = rng.choice(["treatment", "no_intervention"], size=n)
        treat_series = pd.Series(treat)

        # Ground truth logistic DGP
        logit0 = 0.5 * f1
        logit1 = 0.5 * f1 - 0.60
        p0 = 1.0 / (1.0 + np.exp(-logit0))
        p1 = 1.0 / (1.0 + np.exp(-logit1))
        true_mean_tau = float(np.mean(p0 - p1))

        p_churn = np.where(treat == "treatment", p1, p0)
        y = pd.Series((rng.uniform(size=n) < p_churn).astype(int))

        learner = TLearnerUplift(base_estimator=LogisticRegression())
        learner.fit(X, treat_series, y)

        # Predict uplift for treatment: should recover true_mean_tau
        pred_tau = learner.predict_uplift(X, "treatment")
        mean_estimated_tau = float(np.mean(pred_tau))

        # Expected recovery within +/- 0.03 tolerance
        self.assertAlmostEqual(mean_estimated_tau, true_mean_tau, delta=0.03)

    def test_x_learner_binary_treatment(self):
        rng = np.random.RandomState(42)
        n = 800
        X = pd.DataFrame({
            "f1": rng.normal(0, 1, size=n),
            "f2": rng.uniform(10, 50, size=n),
        })
        is_treated = pd.Series(rng.choice([True, False], size=n))
        p = np.where(is_treated, 0.25, 0.45)
        y = pd.Series((rng.uniform(size=n) < p).astype(int))

        x_learner = XLearnerUplift()
        x_learner.fit(X, is_treated, y)
        pred_uplift = x_learner.predict_uplift(X)

        self.assertEqual(len(pred_uplift), n)
        # Expected uplift ~ 0.20
        self.assertGreater(np.mean(pred_uplift), 0.05)

    def test_qini_and_auuc_computation(self):
        n = 200
        y_churn = np.array([0, 1] * (n // 2))
        is_treated = np.array([1, 1, 0, 0] * (n // 4))
        uplift_preds = np.linspace(0.2, -0.1, n)

        qini_res = compute_qini_curve(y_churn, is_treated, uplift_preds, n_points=5)
        self.assertIn("qini_gains", qini_res)
        self.assertIn("random_gains", qini_res)
        self.assertEqual(len(qini_res["fractions"]), 5)

        auuc = compute_auuc(qini_res["fractions"], qini_res["qini_gains"])
        self.assertIsInstance(auuc, float)


if __name__ == "__main__":
    unittest.main()

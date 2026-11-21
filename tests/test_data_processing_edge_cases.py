"""Tests for edge cases in data processing: empty DataFrames, missing columns, constant/all-same values."""
import unittest
import numpy as np
import pandas as pd

from src.features import compute_engineered_features, build_feature_dataframe, score_customer_records
from src.feature_engineering import add_features
from dashboard.core.clv import calculate_customer_clv, calculate_discount_factor, CLVParameters
from dashboard.core.uplift_simulator import run_randomised_pilot, compute_qini_curve
from dashboard.core.actions import load_action_catalog


class TestDataProcessingEdgeCases(unittest.TestCase):
    def setUp(self):
        self.required_cols = [
            "CustomerId", "CreditScore", "Geography", "Gender", "Age",
            "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
            "EstimatedSalary", "TransactionCount", "OnlineBankingUsage",
            "CustomerComplaints", "LoanStatus", "LoanAmount"
        ]

    def _create_sample_row(self, **kwargs):
        base = {
            "CustomerId": 12345,
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Male",
            "Age": 40,
            "Tenure": 3,
            "Balance": 50000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 60000.0,
            "TransactionCount": 15,
            "OnlineBankingUsage": 45.0,
            "CustomerComplaints": 0,
            "LoanStatus": "Approved",
            "LoanAmount": 10000.0,
        }
        base.update(kwargs)
        return base

    def test_compute_features_empty_dataframe(self):
        """Empty DataFrame with valid columns returns an empty DataFrame with all engineered columns."""
        empty_df = pd.DataFrame(columns=self.required_cols)
        result = compute_engineered_features(empty_df)
        self.assertEqual(len(result), 0)
        for col in ["TransactionFrequency", "AccountAgeScore", "LoanToBalanceRatio", "DigitalUsageScore", "CustomerActivityScore"]:
            self.assertIn(col, result.columns)

    def test_compute_features_all_zeros(self):
        """Values of zero across all numeric attributes should not trigger ZeroDivisionError."""
        zero_row = {col: 0 for col in self.required_cols}
        zero_row["Geography"] = "France"
        zero_row["Gender"] = "Female"
        zero_row["LoanStatus"] = "Approved"
        df = pd.DataFrame([zero_row])

        res = compute_engineered_features(df)
        self.assertEqual(len(res), 1)
        self.assertFalse(res["TransactionFrequency"].isna().any())
        self.assertFalse(res["AccountAgeScore"].isna().any())
        self.assertFalse(res["LoanToBalanceRatio"].isna().any())
        self.assertFalse(res["DigitalUsageScore"].isna().any())
        self.assertFalse(res["CustomerActivityScore"].isna().any())
        self.assertEqual(res["TransactionFrequency"].iloc[0], 0.0)
        self.assertEqual(res["LoanToBalanceRatio"].iloc[0], 0.0)

    def test_compute_features_boundary_age_and_tenure(self):
        """Age = 18 and Tenure = 0 should handle denominator clipping without NaN."""
        row = self._create_sample_row(Age=18, Tenure=0, Balance=0.0, LoanAmount=0.0)
        df = pd.DataFrame([row])
        res = compute_engineered_features(df)
        self.assertEqual(res["AccountAgeScore"].iloc[0], 0.0)
        self.assertEqual(res["LoanToBalanceRatio"].iloc[0], 0.0)

    def test_build_feature_dataframe_missing_optional_columns(self):
        """If optional features are absent, build_feature_dataframe fills defaults gracefully."""
        row = {
            "CreditScore": 650,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 50,
            "Tenure": 5,
            "Balance": 80000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 90000.0,
        }
        df = pd.DataFrame([row])
        X = build_feature_dataframe(df)
        self.assertEqual(len(X), 1)
        self.assertIn("CustomerComplaints", X.columns)
        self.assertIn("TransactionFrequency", X.columns)
        self.assertIn("CustomerActivityScore", X.columns)
        self.assertEqual(X["CustomerComplaints"].iloc[0], 0.0)

    def test_score_customer_records_all_same_values(self):
        """Scoring 100 identical customers produces deterministic, equal calibrated probabilities."""
        row = self._create_sample_row()
        df = pd.DataFrame([row] * 100)
        probs = score_customer_records(df)

        self.assertEqual(len(probs), 100)
        # All probabilities should be identical
        self.assertTrue(np.allclose(probs, probs[0]))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))

    def test_clv_calculation_edge_cases(self):
        """CLV calculations handle 0 balance, high balances, and boundary discount rates."""
        params = CLVParameters(
            nim=0.03,
            fee_per_product=50.0,
            servicing_cost=40.0,
            horizon_years=3,
            discount_rate=0.08,
        )
        # Zero balance customer with positive products
        clv_zero_bal = calculate_customer_clv(balance=0.0, num_products=1, params=params)
        self.assertGreaterEqual(clv_zero_bal, params.min_clv_floor)

        # High balance customer
        clv_high_bal = calculate_customer_clv(balance=250000.0, num_products=3, params=params)
        self.assertGreater(clv_high_bal, clv_zero_bal)

        # Discount factor calculation
        dfactor = calculate_discount_factor(horizon_years=3, discount_rate=0.08)
        self.assertGreater(dfactor, 0.0)
        self.assertLess(dfactor, 3.0)

    def test_uplift_pilot_simulation_edge_cases(self):
        """Randomised pilot simulation handles uniform customer inputs cleanly."""
        catalog = load_action_catalog()
        rows = [self._create_sample_row(Balance=10000.0, Age=30) for _ in range(10)]
        df = pd.DataFrame(rows)
        df["CalibratedProbability"] = 0.40

        pilot = run_randomised_pilot(df, catalog=catalog, sample_size=10, seed=42)
        self.assertEqual(len(pilot.data), 10)
        self.assertTrue(pilot.is_simulated)
        self.assertIn("ObservedChurn", pilot.data.columns)
        self.assertFalse(pilot.data["ObservedChurn"].isna().any())


if __name__ == "__main__":
    unittest.main()

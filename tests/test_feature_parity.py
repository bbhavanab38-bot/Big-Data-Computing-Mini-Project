"""Test suite proving zero training-serving skew between batch and streaming feature pipelines."""
import unittest
import numpy as np
import pandas as pd

from src import config
from src.feature_engineering import add_features
from src.features import (
    compute_engineered_features,
    build_feature_dataframe,
    score_customer_records,
)


class TestFeatureParity(unittest.TestCase):
    def setUp(self):
        # Sample customer records
        self.sample_records = [
            {
                "CustomerId": 15601539,
                "CreditScore": 619,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 2,
                "Balance": 0.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 101348.88,
                "TransactionCount": 15,
                "OnlineBankingUsage": 45,
                "CustomerComplaints": 0,
                "LoanAmount": 0.0,
                "LoanStatus": "No Loan",
            },
            {
                "CustomerId": 15606332,
                "CreditScore": 708,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 52,
                "Tenure": 8,
                "Balance": 125000.50,
                "NumOfProducts": 2,
                "HasCrCard": 0,
                "IsActiveMember": 0,
                "EstimatedSalary": 145000.00,
                "TransactionCount": 42,
                "OnlineBankingUsage": 80,
                "CustomerComplaints": 2,
                "LoanAmount": 15000.0,
                "LoanStatus": "Active Loan",
            },
        ]
        self.batch_df = pd.DataFrame(self.sample_records)

    def test_engineered_feature_formulas_identical(self):
        """Test that compute_engineered_features produces identical results to add_features."""
        feat_direct = compute_engineered_features(self.batch_df)
        feat_legacy = add_features(self.batch_df)

        for col in config.ENGINEERED:
            np.testing.assert_allclose(
                feat_direct[col].values,
                feat_legacy[col].values,
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"Discrepancy detected in engineered feature '{col}'",
            )

    def test_streaming_single_row_matches_batch_scoring(self):
        """Test that single-record streaming scoring produces identical probabilities to batch scoring."""
        # Batch scoring of 2 records
        batch_probs = score_customer_records(self.batch_df)

        # Streaming single-record scoring
        stream_probs = []
        for rec in self.sample_records:
            single_df = pd.DataFrame([rec])
            p = score_customer_records(single_df)[0]
            stream_probs.append(p)

        np.testing.assert_allclose(
            batch_probs,
            stream_probs,
            rtol=1e-9,
            atol=1e-9,
            err_msg="Training/serving skew detected: batch and streaming probabilities differ!",
        )

    def test_missing_field_default_imputation(self):
        """Test that optional missing fields are imputed with documented defaults."""
        partial_record = [{
            "CustomerId": 15609999,
            "CreditScore": 680,
            "Age": 45,
            # Balance, NumOfProducts, etc., omitted
        }]
        feat_df = build_feature_dataframe(partial_record)
        self.assertEqual(len(feat_df), 1)
        for expected_col in config.MODEL_FEATURES:
            self.assertIn(expected_col, feat_df.columns)
            self.assertFalse(pd.isna(feat_df[expected_col].iloc[0]))


if __name__ == "__main__":
    unittest.main()

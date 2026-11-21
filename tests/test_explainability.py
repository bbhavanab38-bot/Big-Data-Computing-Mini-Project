"""Unit tests for dashboard.core.explainability (customer explanation & permutation importance)."""
import unittest
from pathlib import Path
import joblib
import pandas as pd

from dashboard.core.explainability import (
    clean_feature_name,
    compute_permutation_importance,
    explain_customer_prediction,
)

BASE_DIR = Path(__file__).resolve().parents[1]
TEST_FILE = BASE_DIR / "data" / "processed" / "test_split.csv"
BEST_MODEL_FILE = BASE_DIR / "models" / "best_model.pkl"


class TestExplainability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_df = pd.read_csv(TEST_FILE)
        cls.bundle = joblib.load(BEST_MODEL_FILE)

    def test_clean_feature_name(self):
        self.assertEqual(clean_feature_name("num__Balance"), "Balance")
        self.assertEqual(clean_feature_name("bin__HasCrCard"), "HasCrCard")
        self.assertEqual(clean_feature_name("cat__Geography_Germany"), "Geography: Germany")

    def test_explain_customer_prediction(self):
        customer = self.test_df.iloc[0]
        res = explain_customer_prediction(customer, self.bundle)
        self.assertIn("customer_id", res)
        self.assertIn("churn_probability", res)
        self.assertIn("risk_band", res)
        self.assertIn(res["risk_band"], ["Low", "Medium", "High"])
        self.assertGreaterEqual(res["churn_probability"], 0.0)
        self.assertLessEqual(res["churn_probability"], 1.0)

        # Check factors
        self.assertIn("factors_increasing_risk", res)
        self.assertIn("factors_reducing_risk", res)
        self.assertIsInstance(res["factors_increasing_risk"], list)
        self.assertIsInstance(res["factors_reducing_risk"], list)

    def test_compute_permutation_importance(self):
        small_test = self.test_df.head(100)
        imp = compute_permutation_importance(self.bundle, small_test, n_repeats=2, random_state=42)
        self.assertIn("Feature", imp.columns)
        self.assertIn("ImportanceMean", imp.columns)
        self.assertIn("ImportanceStd", imp.columns)
        self.assertEqual(len(imp), len(self.bundle["features"]))


if __name__ == "__main__":
    unittest.main()


"""Unit tests ensuring zero data leakage in out-of-fold scored customer dataset."""
import unittest
import pandas as pd
from src import config


class TestOOFLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scored_file = config.SCORED_DATASET_FILE
        cls.test_file = config.TEST_FILE

    def test_scored_dataset_exists_and_complete(self):
        self.assertTrue(self.scored_file.exists(), f"Missing {self.scored_file}")
        df = pd.read_csv(self.scored_file)
        self.assertEqual(len(df), 10000, "Scored dataset must contain all 10,000 customers")

        required_cols = ["CustomerId", "CalibratedProbability", "ScoreSource", "model_version", "Segment", "RiskCategory", "PredictedChurn"]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Column {col} missing in scored dataset")
            self.assertFalse(df[col].isna().any(), f"Column {col} contains null values")

    def test_score_source_partitioning(self):
        df = pd.read_csv(self.scored_file)
        sources = set(df["ScoreSource"].unique())
        self.assertEqual(sources, {"out_of_fold", "test"})

        counts = df["ScoreSource"].value_counts()
        self.assertEqual(counts["out_of_fold"], 8000, "Exactly 8,000 training rows must be scored out-of-fold")
        self.assertEqual(counts["test"], 2000, "Exactly 2,000 test rows must be scored from test holdout")

        # Unique customer IDs
        self.assertEqual(df["CustomerId"].nunique(), 10000, "All customer IDs must be distinct")

    def test_test_split_alignment(self):
        df_scored = pd.read_csv(self.scored_file)
        df_test = pd.read_csv(self.test_file)

        test_scored_ids = set(df_scored[df_scored["ScoreSource"] == "test"]["CustomerId"])
        holdout_ids = set(df_test["CustomerId"])

        self.assertEqual(test_scored_ids, holdout_ids, "Test split IDs must match test score source exactly")

    def test_probability_bounds_and_risk_mapping(self):
        df = pd.read_csv(self.scored_file)
        probs = df["CalibratedProbability"]
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

        # Risk categories must match config thresholds
        for _, row in df.sample(n=100, random_state=42).iterrows():
            expected_risk = config.get_risk_category(row["CalibratedProbability"])
            self.assertEqual(row["RiskCategory"], expected_risk)


if __name__ == "__main__":
    unittest.main()


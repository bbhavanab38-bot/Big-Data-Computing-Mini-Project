"""Unit tests verifying segment naming stability under cluster index permutations."""
import unittest
import pandas as pd
from dashboard.core.segments import (
    build_segment_profiles,
    derive_segment_metadata_by_rules,
    assign_kmeans_segments,
    SEGMENT_FEATURE_COLS,
)


class TestSegmentStability(unittest.TestCase):
    def setUp(self):
        # Base cluster profiles matching Spark summary metrics
        self.base_summary = pd.DataFrame([
            {"Segment": 0, "Customers": 2712, "AvgAge": 39.6, "AvgBalance": 71662.0, "AvgActivity": 0.261, "ChurnRatePct": 22.64},
            {"Segment": 1, "Customers": 2471, "AvgAge": 34.1, "AvgBalance": 58438.0, "AvgActivity": 0.666, "ChurnRatePct": 7.69},
            {"Segment": 2, "Customers": 2535, "AvgAge": 35.5, "AvgBalance": 50094.0, "AvgActivity": 0.342, "ChurnRatePct": 16.84},
            {"Segment": 3, "Customers": 2282, "AvgAge": 47.5, "AvgBalance": 87999.0, "AvgActivity": 0.630, "ChurnRatePct": 18.01},
        ])

    def test_original_spark_alignment(self):
        profiles = build_segment_profiles(self.base_summary)
        names = dict(zip(profiles["Segment"], profiles["SegmentName"]))
        self.assertEqual(names[0], "Inactive High-Balance Disengaged")
        self.assertEqual(names[1], "Young High-Activity Loyalists")
        self.assertEqual(names[2], "Entry-Level Moderate-Risk")
        self.assertEqual(names[3], "Affluent Mature Engaged")

    def test_permutation_invariance(self):
        # Permute cluster IDs: e.g. reverse segment IDs: 0->3, 1->2, 2->1, 3->0
        permuted = self.base_summary.copy()
        permuted["Segment"] = [3, 2, 1, 0]
        # Re-sort to simulate arbitrary cluster output order
        permuted = permuted.sample(frac=1.0, random_state=123).reset_index(drop=True)

        profiles = build_segment_profiles(permuted)
        names = dict(zip(profiles["Segment"], profiles["SegmentName"]))

        # Segment 3 now holds the metrics of original segment 0 (highest churn, low activity)
        self.assertEqual(names[3], "Inactive High-Balance Disengaged")
        # Segment 2 holds metrics of original segment 1 (lowest churn, high activity)
        self.assertEqual(names[2], "Young High-Activity Loyalists")
        # Segment 1 holds metrics of original segment 2 (lower balance)
        self.assertEqual(names[1], "Entry-Level Moderate-Risk")
        # Segment 0 holds metrics of original segment 3 (affluent older)
        self.assertEqual(names[0], "Affluent Mature Engaged")

    def test_assign_kmeans_segments_customer_level(self):
        # Create a synthetic mini dataset to test customer-level assignment
        mini_data = pd.DataFrame({
            "CustomerId": range(1001, 1101),
            "Age": [30, 45, 25, 55] * 25,
            "Balance": [20000, 120000, 5000, 95000] * 25,
            "CreditScore": [650, 700, 600, 720] * 25,
            "TransactionCount": [15, 5, 25, 30] * 25,
            "EstimatedSalary": [50000, 100000, 40000, 120000] * 25,
            "CustomerActivityScore": [0.6, 0.2, 0.7, 0.5] * 25,
            "Exited": [0, 1, 0, 0] * 25,
        })
        df_assigned, profiles = assign_kmeans_segments(mini_data, k=4, random_state=42)
        self.assertIn("Segment", df_assigned.columns)
        self.assertIn("SegmentName", df_assigned.columns)
        self.assertEqual(len(df_assigned), 100)
        self.assertEqual(len(profiles), 4)
        self.assertEqual(set(df_assigned["Segment"]), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()


"""Unit tests for dashboard.core.segments (K-Means segment profiling and naming)."""
import unittest
from pathlib import Path
import pandas as pd

from dashboard.core.data import load_spark_export
from dashboard.core.segments import (
    DERIVED_SEGMENT_DEFINITIONS,
    build_segment_profiles,
    get_normalized_radar_metrics,
)

BASE_DIR = Path(__file__).resolve().parents[1]
SPARK_DIR = BASE_DIR / "reports" / "spark"


class TestSegments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_segments = load_spark_export(SPARK_DIR, "segment_churn_summary")

    def test_segment_definitions(self):
        self.assertIn(0, DERIVED_SEGMENT_DEFINITIONS)
        self.assertIn(1, DERIVED_SEGMENT_DEFINITIONS)
        self.assertIn(2, DERIVED_SEGMENT_DEFINITIONS)
        self.assertIn(3, DERIVED_SEGMENT_DEFINITIONS)

        # Segment 0 has highest churn
        self.assertEqual(DERIVED_SEGMENT_DEFINITIONS[0]["name"], "Inactive High-Balance Disengaged")
        # Segment 1 has lowest churn & highest activity
        self.assertEqual(DERIVED_SEGMENT_DEFINITIONS[1]["name"], "Young High-Activity Loyalists")

    def test_build_segment_profiles(self):
        self.assertIsNotNone(self.raw_segments)
        profiles = build_segment_profiles(self.raw_segments)
        self.assertEqual(len(profiles), 4)
        self.assertIn("SegmentName", profiles.columns)
        self.assertIn("CustomerSharePct", profiles.columns)
        self.assertIn("RiskLevel", profiles.columns)
        self.assertIn("RecommendedStrategy", profiles.columns)
        self.assertAlmostEqual(profiles["CustomerSharePct"].sum(), 100.0, places=1)

    def test_get_normalized_radar_metrics(self):
        profiles = build_segment_profiles(self.raw_segments)
        radar = get_normalized_radar_metrics(profiles)
        self.assertFalse(radar.empty)
        self.assertIn("NormalizedScore", radar.columns)
        self.assertTrue((radar["NormalizedScore"] >= 0.0).all())
        self.assertTrue((radar["NormalizedScore"] <= 1.0).all())


if __name__ == "__main__":
    unittest.main()


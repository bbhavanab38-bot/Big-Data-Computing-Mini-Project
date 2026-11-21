"""Unit tests for executive reporting, PDF generation, and CSV exports."""
import unittest
import pandas as pd
from dashboard.core.reports import (
    build_report_data,
    generate_allocations_csv,
    generate_pdf_report,
    generate_summary_csv,
)


class TestReportsEngine(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame([
            {
                "CustomerId": 1001,
                "CalibratedProbability": 0.72,
                "RiskCategory": "High",
                "SegmentName": "High Balance At-Risk",
                "Balance": 125000.0,
                "EstimatedCLV": 3400.0,
                "Exited": 1,
            },
            {
                "CustomerId": 1002,
                "CalibratedProbability": 0.12,
                "RiskCategory": "Low",
                "SegmentName": "Low Balance Retained",
                "Balance": 8000.0,
                "EstimatedCLV": 1200.0,
                "Exited": 0,
            },
            {
                "CustomerId": 1003,
                "CalibratedProbability": 0.38,
                "RiskCategory": "Medium",
                "SegmentName": "Dormant Modest-Wealth",
                "Balance": 45000.0,
                "EstimatedCLV": 2100.0,
                "Exited": 0,
            },
        ])

    def test_build_report_data_structure(self):
        """Report data must contain all core metadata, model, portfolio, and simulation keys."""
        data = build_report_data(scored_df=self.sample_df, action_budget=15000.0, decision_threshold=0.40)
        
        required_keys = [
            "generated_at", "platform_name", "platform_version", "model_version",
            "model_name", "calibration_method", "training_data_hash", "metrics",
            "total_customers", "baseline_churn", "mean_calibrated_prob", "decision_threshold",
            "risk_tiers", "segments", "actions", "action_budget",
            "simulated_interventions_count", "simulated_prevented_churns",
            "simulated_gross_clv_saved", "simulated_net_roi", "fairness",
        ]
        for k in required_keys:
            self.assertIn(k, data, f"Missing key in report data: {k}")

        self.assertEqual(data["total_customers"], 3)
        self.assertEqual(data["action_budget"], 15000.0)
        self.assertEqual(data["decision_threshold"], 0.40)
        self.assertIn("low", data["risk_tiers"])
        self.assertIn("high", data["risk_tiers"])

    def test_generate_pdf_report_validity(self):
        """PDF generation must return a valid non-empty PDF byte stream."""
        data = build_report_data(scored_df=self.sample_df)
        pdf_bytes = generate_pdf_report(data)

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 2000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF-"), "Generated file is not a valid PDF")

    def test_generate_summary_csv(self):
        """Summary CSV must contain structured category, metric, and value rows."""
        data = build_report_data(scored_df=self.sample_df)
        csv_str = generate_summary_csv(data)

        self.assertIsInstance(csv_str, str)
        lines = csv_str.strip().splitlines()
        self.assertGreater(len(lines), 15)
        self.assertEqual(lines[0], "Category,Metric,Value")
        
        # Check that key sections exist
        self.assertTrue(any("System Metadata" in line for line in lines))
        self.assertTrue(any("Portfolio Overview" in line for line in lines))
        self.assertTrue(any("Model Performance" in line for line in lines))
        self.assertTrue(any("Algorithmic Fairness" in line for line in lines))

    def test_generate_allocations_csv(self):
        """Allocations CSV must output prioritized at-risk customers with assigned actions."""
        alloc_csv = generate_allocations_csv(self.sample_df, limit=10)
        self.assertIsInstance(alloc_csv, str)
        lines = alloc_csv.strip().splitlines()
        self.assertEqual(len(lines), 4)  # 1 header + 3 rows
        
        header = lines[0]
        self.assertIn("CustomerId", header)
        self.assertIn("CalibratedProbability", header)
        self.assertIn("RecommendedAction", header)
        self.assertIn("EstimatedCLVEUR", header)

        # Sorted by risk descending (Customer 1001 with 0.72 should be first data line)
        self.assertIn("1001", lines[1])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for dashboard.core.report_generator (HTML Executive Summary)."""
import unittest
import pandas as pd

from dashboard.core.report_generator import generate_executive_report_html


class TestReportGenerator(unittest.TestCase):
    def test_generate_executive_report_html(self):
        kpis = {
            "total": 10000,
            "churned": 2037,
            "retained": 7963,
            "churn_rate": 20.37,
            "avg_probability": 0.43,
        }
        py_models = pd.DataFrame([{
            "Model": "Logistic Regression",
            "Accuracy": 0.70,
            "Precision": 0.35,
            "Recall": 0.70,
            "F1 Score": 0.47,
            "ROC-AUC": 0.76,
        }])
        spark_models = pd.DataFrame([{
            "Model": "Logistic Regression",
            "Accuracy": 0.66,
            "ChurnPrecision": 0.27,
            "ChurnRecall": 0.65,
            "ChurnF1": 0.38,
            "ROC_AUC": 0.72,
        }])
        segments = pd.DataFrame([{
            "Segment": 0,
            "SegmentName": "Inactive High-Balance Disengaged",
            "Customers": 2712,
            "CustomerSharePct": 27.1,
            "AvgAge": 39.6,
            "AvgBalance": 71662.0,
            "AvgActivity": 0.261,
            "ChurnRatePct": 22.64,
            "RiskLevel": "Critical",
        }])
        drivers = pd.DataFrame([{"Feature": "Age", "ImportanceMean": 0.048}])

        html = generate_executive_report_html(
            kpis=kpis,
            python_models=py_models,
            spark_models=spark_models,
            segment_profiles=segments,
            top_drivers=drivers,
            high_risk_count=500,
        )

        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("10,000 synthetic customer records", html)
        self.assertIn("Inactive High-Balance Disengaged", html)
        self.assertIn("20.37%", html)
        self.assertIn("Track A: Python / scikit-learn", html)
        self.assertIn("Track B: Spark MLlib on YARN", html)
        self.assertIn("@media print", html)


if __name__ == "__main__":
    unittest.main()


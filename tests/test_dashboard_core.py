"""Unit tests for dashboard.core.data (pure logic, no Streamlit dependency)."""
import unittest
from pathlib import Path
import pandas as pd

from dashboard.core.data import (
    INTEGER_COLUMNS,
    MODEL_METRICS,
    load_data,
    load_spark_export,
    filter_customers,
    calculate_overview_kpis,
    prepare_geo_churn,
    prepare_age_churn,
    prepare_status_counts,
    prepare_risk_counts,
    build_confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "reports" / "powerbi"
SPARK_DIR = BASE_DIR / "reports" / "spark"


class TestDashboardCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.customers, cls.high_risk, cls.models = load_data(
            DATA_DIR / "powerbi_dataset.csv",
            DATA_DIR / "high_risk_customers.csv",
            DATA_DIR / "model_comparison.csv",
        )

    def test_load_data(self):
        self.assertFalse(self.customers.empty)
        self.assertFalse(self.high_risk.empty)
        self.assertFalse(self.models.empty)
        self.assertIn("CustomerId", self.customers.columns)
        self.assertIn("Exited", self.customers.columns)
        self.assertIn("ChurnProbability", self.customers.columns)
        self.assertIn("Model", self.models.columns)

    def test_load_spark_export(self):
        df_all = load_spark_export(SPARK_DIR, "spark_model_comparison_validated_all")
        self.assertIsNotNone(df_all)
        self.assertGreaterEqual(len(df_all), 4)
        self.assertIn("Model", df_all.columns)
        self.assertIn("ROC_AUC", df_all.columns)

        missing = load_spark_export(SPARK_DIR, "non_existent_folder")
        self.assertIsNone(missing)

    def test_filter_customers(self):
        filtered = filter_customers(
            self.customers,
            geographies=["France"],
            genders=["Female"],
            risk_segments=["High"],
        )
        self.assertTrue((filtered["Geography"] == "France").all())
        self.assertTrue((filtered["Gender"] == "Female").all())
        self.assertTrue((filtered["RiskSegment"] == "High").all())

    def test_calculate_overview_kpis(self):
        kpis = calculate_overview_kpis(self.customers)
        self.assertEqual(kpis["total"], len(self.customers))
        self.assertEqual(kpis["churned"] + kpis["retained"], kpis["total"])
        self.assertGreaterEqual(kpis["churn_rate"], 0.0)
        self.assertLessEqual(kpis["churn_rate"], 100.0)
        self.assertGreaterEqual(kpis["avg_probability"], 0.0)
        self.assertLessEqual(kpis["avg_probability"], 1.0)

    def test_calculate_overview_kpis_empty(self):
        empty_df = pd.DataFrame(columns=["Exited", "ChurnProbability"])
        kpis = calculate_overview_kpis(empty_df)
        self.assertEqual(kpis["total"], 0)
        self.assertEqual(kpis["churn_rate"], 0.0)

    def test_prepare_geo_churn(self):
        geo = prepare_geo_churn(self.customers)
        self.assertIn("Geography", geo.columns)
        self.assertIn("Customers", geo.columns)
        self.assertIn("ChurnRate", geo.columns)
        self.assertEqual(geo["Customers"].sum(), len(self.customers))

    def test_prepare_age_churn(self):
        age = prepare_age_churn(self.customers)
        self.assertIn("AgeBand", age.columns)
        self.assertIn("ChurnRate", age.columns)

    def test_prepare_status_counts(self):
        status = prepare_status_counts(self.customers)
        self.assertIn("Status", status.columns)
        self.assertIn("Customers", status.columns)
        self.assertEqual(status["Customers"].sum(), len(self.customers))

    def test_prepare_risk_counts(self):
        risk = prepare_risk_counts(self.customers)
        self.assertIn("Risk", risk.columns)
        self.assertIn("Customers", risk.columns)
        self.assertEqual(risk["Customers"].sum(), len(self.customers))

    def test_build_confusion_matrix(self):
        row = pd.Series({"TP": 100, "FP": 20, "TN": 800, "FN": 80})
        cm = build_confusion_matrix(row)
        self.assertEqual(cm.loc["Actual Exited=0", "Predicted 0"], 800)
        self.assertEqual(cm.loc["Actual Exited=0", "Predicted 1"], 20)
        self.assertEqual(cm.loc["Actual Exited=1", "Predicted 0"], 80)
        self.assertEqual(cm.loc["Actual Exited=1", "Predicted 1"], 100)


if __name__ == "__main__":
    unittest.main()


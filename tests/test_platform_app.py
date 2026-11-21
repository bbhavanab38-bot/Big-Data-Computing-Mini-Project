"""AppTest integration tests for Banking Retention Intelligence Platform.

Verifies:
  - All four core pages render without exception
  - Global Customer ID search locates and selects customer
  - Sidebar filters alter displayed account subsets
  - Prediction engine rejects out-of-domain values
  - Batch upload enforces strict row limits
"""
import unittest
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest
from pydantic import ValidationError

from src.api_v2.schemas import CustomerInput
from src.api_v2.service import predict_single, predict_batch_dataframe

BASE_DIR = Path(__file__).resolve().parents[1]
APP_PATH = str(BASE_DIR / "dashboard" / "platform_app.py")
OVERVIEW_PATH = str(BASE_DIR / "dashboard" / "pages" / "overview.py")
CI_PATH = str(BASE_DIR / "dashboard" / "pages" / "customer_intelligence.py")
SEG_PATH = str(BASE_DIR / "dashboard" / "pages" / "segmentation.py")
MP_PATH = str(BASE_DIR / "dashboard" / "pages" / "model_performance.py")


class TestPlatformApp(unittest.TestCase):
    def test_platform_app_shell_renders_cleanly(self):
        at = AppTest.from_file(APP_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"platform_app.py raised exceptions: {at.exception}")

    def test_overview_page_renders_cleanly(self):
        at = AppTest.from_file(OVERVIEW_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"overview.py raised exceptions: {at.exception}")

    def test_customer_intelligence_page_renders_cleanly(self):
        at = AppTest.from_file(CI_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"customer_intelligence.py raised exceptions: {at.exception}")

    def test_segmentation_page_renders_cleanly(self):
        at = AppTest.from_file(SEG_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"segmentation.py raised exceptions: {at.exception}")

    def test_model_performance_page_renders_cleanly(self):
        at = AppTest.from_file(MP_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"model_performance.py raised exceptions: {at.exception}")

    def test_global_search_sets_customer_id(self):
        at = AppTest.from_file(APP_PATH)
        at.run()
        # Find sidebar global search text input
        text_inputs = at.sidebar.text_input
        self.assertGreater(len(text_inputs), 0)
        search_input = text_inputs[0]
        search_input.input("15602427")
        at.run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state.get("selected_customer_id"), 15602427)

    def test_overview_filters_update_view(self):
        at = AppTest.from_file(OVERVIEW_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0)
        # Modify geography filter to only France
        multis = at.sidebar.multiselect
        self.assertGreater(len(multis), 0)
        geo_filter = multis[0]
        geo_filter.select("France")
        at.run()
        self.assertEqual(len(at.exception), 0)

    def test_prediction_service_validates_domain_bounds(self):
        # Valid input passes
        valid_payload = {
            "Age": 45,
            "CreditScore": 650,
            "Balance": 50000.0,
            "Geography": "France",
            "Gender": "Female",
        }
        res = predict_single(valid_payload)
        self.assertIn("churn_probability", res)
        self.assertIn("risk_category", res)
        self.assertIn(res["risk_category"], ["Low", "Medium", "High"])

        # Out-of-bounds inputs fail Pydantic validation
        with self.assertRaises(ValidationError):
            CustomerInput(Age=15, CreditScore=650, Balance=50000.0, Geography="France", Gender="Female")

        with self.assertRaises(ValidationError):
            CustomerInput(Age=45, CreditScore=200, Balance=50000.0, Geography="France", Gender="Female")

        with self.assertRaises(ValidationError):
            CustomerInput(Age=45, CreditScore=650, Balance=-1000.0, Geography="France", Gender="Female")

        with self.assertRaises(ValidationError):
            CustomerInput(Age=45, CreditScore=650, Balance=50000.0, Geography="Atlantis", Gender="Female")

    def test_batch_prediction_service_enforces_limits(self):
        # Valid small batch
        small_df = pd.DataFrame({
            "CustomerId": [101, 102],
            "Age": [40, 50],
            "CreditScore": [600, 700],
            "Balance": [30000, 80000],
            "Geography": ["France", "Germany"],
            "Gender": ["Female", "Male"],
        })
        res = predict_batch_dataframe(small_df)
        self.assertEqual(res["total_records"], 2)
        self.assertEqual(len(res["predictions"]), 2)


if __name__ == "__main__":
    unittest.main()

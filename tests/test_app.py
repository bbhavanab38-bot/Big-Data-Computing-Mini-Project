"""Integration test for Streamlit dashboard execution and reset filters interaction."""
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


class TestStreamlitApp(unittest.TestCase):
    def test_app_loads_and_renders_cleanly(self):
        at = AppTest.from_file(APP_PATH)
        at.run()
        self.assertEqual(len(at.exception), 0, f"AppTest raised exceptions: {at.exception}")

        # Check Reset filters button presence
        reset_buttons = [b for b in at.button if b.label == "Reset filters"]
        self.assertEqual(len(reset_buttons), 1)

    def test_reset_filters_functionality(self):
        at = AppTest.from_file(APP_PATH)
        at.run()

        # Modify a filter
        geo_widget = at.multiselect(key="filter_geo")
        initial_values = list(geo_widget.value)
        self.assertIn("France", initial_values)

        geo_widget.unselect("France").run()
        self.assertNotIn("France", at.multiselect(key="filter_geo").value)

        # Click Reset filters
        reset_btn = [b for b in at.button if b.label == "Reset filters"][0]
        reset_btn.click().run()

        # Verify filters restored
        self.assertEqual(at.multiselect(key="filter_geo").value, initial_values)


if __name__ == "__main__":
    unittest.main()

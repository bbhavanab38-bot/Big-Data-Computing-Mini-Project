"""Integration Tests for Phase 4 Streamlit Pages via AppTest."""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parents[1]
DECISION_ENGINE_PATH = str(BASE_DIR / "dashboard" / "pages" / "decision_engine.py")
EXPLAINABILITY_PATH = str(BASE_DIR / "dashboard" / "pages" / "explainability.py")
ROI_SIMULATOR_PATH = str(BASE_DIR / "dashboard" / "pages" / "roi_simulator.py")
PLATFORM_APP_PATH = str(BASE_DIR / "dashboard" / "platform_app.py")


class TestPlatformPhase4Pages(unittest.TestCase):
    """Test suite verifying Streamlit page rendering and widget interactions for Phase 4."""

    def test_decision_engine_page_renders_cleanly(self):
        at = AppTest.from_file(DECISION_ENGINE_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0, f"decision_engine.py raised: {at.exception}")
        # Verify simulation banner is present
        banners = [m.value for m in at.markdown if "Simulated uplift" in m.value or "Causal Simulation" in m.value]
        self.assertGreater(len(banners), 0)

    def test_explainability_page_renders_cleanly(self):
        at = AppTest.from_file(EXPLAINABILITY_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0, f"explainability.py raised: {at.exception}")
        # Verify SHAP notice is present
        notices = [m.value for m in at.markdown if "Model Interpretability Notice" in m.value]
        self.assertGreater(len(notices), 0)

    def test_roi_simulator_page_renders_cleanly(self):
        at = AppTest.from_file(ROI_SIMULATOR_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0, f"roi_simulator.py raised: {at.exception}")
        # Verify planning assumption notice is present
        notices = [m.value for m in at.markdown if "Assumption-Based Planning Model" in m.value]
        self.assertGreater(len(notices), 0)

    def test_decision_engine_optimization_trigger(self):
        at = AppTest.from_file(DECISION_ENGINE_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0)
        # Click Run Optimization button
        if at.button:
            run_btns = [b for b in at.button if "Run Optimization" in getattr(b, "label", "")]
            if run_btns:
                run_btns[0].click().run(timeout=35)
                self.assertEqual(len(at.exception), 0)
                self.assertIn("last_optim_result", at.session_state)

    def test_roi_simulator_scenario_save(self):
        at = AppTest.from_file(ROI_SIMULATOR_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0)
        # Check default scenarios exist in session state
        scenarios = at.session_state.get("saved_roi_scenarios", {})
        self.assertGreater(len(scenarios), 0)


if __name__ == "__main__":
    unittest.main()

"""Integration Tests for Phase 6 Streamlit Pages via AppTest."""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

from src.streaming.event_simulator import run_event_generator
from src.streaming.microbatch_runner import run_microbatch_job

BASE_DIR = Path(__file__).resolve().parents[1]
REALTIME_PAGE_PATH = str(BASE_DIR / "dashboard" / "pages" / "realtime_monitoring.py")
OVERVIEW_PAGE_PATH = str(BASE_DIR / "dashboard" / "pages" / "overview.py")
MODEL_PERF_PATH = str(BASE_DIR / "dashboard" / "pages" / "model_performance.py")


class TestPlatformPhase6Pages(unittest.TestCase):
    """Test suite verifying Streamlit page rendering and widget interactions for Phase 6."""

    @classmethod
    def setUpClass(cls):
        # Generate simulated events and run a micro-batch cycle so telemetry exists for tests
        run_event_generator(events=100, at_risk_ratio=0.20, seed=42)
        run_microbatch_job()

    def test_realtime_monitoring_page_renders_cleanly(self):
        """Verify realtime_monitoring.py renders without exceptions and contains simulated banner."""
        at = AppTest.from_file(REALTIME_PAGE_PATH)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"realtime_monitoring.py raised: {at.exception}")

        # Check simulation notice
        info_boxes = [m.value for m in at.info if "Simulated Event Stream" in m.value]
        self.assertGreater(len(info_boxes), 0, "Expected simulated event stream disclaimer in monitoring page.")

        # Check KPI cards or headers
        headers = [m.value for m in at.markdown if "Real-Time Risk Escalation Watchlist" in m.value or "Streaming Event Telemetry" in m.value]
        self.assertGreater(len(headers), 0)

    def test_overview_page_includes_activity_chart(self):
        """Verify overview.py includes simulated interaction activity over time chart."""
        at = AppTest.from_file(OVERVIEW_PAGE_PATH)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"overview.py raised: {at.exception}")

        # Verify activity section header is present
        activity_mentions = [m.value for m in at.markdown if "Simulated Interaction Activity Over Time" in m.value]
        self.assertGreater(len(activity_mentions), 0, "Expected activity chart section on Overview page.")

    def test_model_performance_includes_drift_monitoring(self):
        """Verify model_performance.py includes model monitoring and drift PSI audit."""
        at = AppTest.from_file(MODEL_PERF_PATH)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"model_performance.py raised: {at.exception}")

        # Verify drift monitoring section header is present
        drift_mentions = [m.value for m in at.markdown if "Population Stability" in m.value or "PSI" in m.value]
        self.assertGreater(len(drift_mentions), 0, "Expected drift monitoring section on Model Performance page.")


if __name__ == "__main__":
    unittest.main()

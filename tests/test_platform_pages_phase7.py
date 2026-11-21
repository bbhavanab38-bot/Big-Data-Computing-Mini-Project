"""Integration Tests for Phase 7 Streamlit Pages via AppTest."""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_PAGE_PATH = str(BASE_DIR / "dashboard" / "pages" / "reports.py")
SETTINGS_PAGE_PATH = str(BASE_DIR / "dashboard" / "pages" / "settings.py")


class TestPlatformPhase7Pages(unittest.TestCase):
    """Test suite verifying Streamlit page rendering and widget interactions for Phase 7."""

    def test_reports_page_renders_cleanly(self):
        """Verify reports.py renders without exceptions and contains download buttons."""
        at = AppTest.from_file(REPORTS_PAGE_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0, f"reports.py raised: {at.exception}")

        # Check warning / legal disclosure
        warnings = [w.value for w in at.warning if "Legal & Simulation Notice" in w.value]
        self.assertGreater(len(warnings), 0, "Expected simulation notice on Reports page.")

        # Check download buttons presence
        dl_buttons = [b.label for b in at.download_button]
        self.assertTrue(any("PDF Audit Briefing" in label for label in dl_buttons))
        self.assertTrue(any("Executive Metrics CSV" in label for label in dl_buttons))
        self.assertTrue(any("Customer Allocations CSV" in label for label in dl_buttons))

    def test_settings_page_renders_cleanly(self):
        """Verify settings.py renders without exceptions and contains health cards."""
        at = AppTest.from_file(SETTINGS_PAGE_PATH)
        at.run(timeout=35)
        self.assertEqual(len(at.exception), 0, f"settings.py raised: {at.exception}")

        # Check headers
        headers = [m.value for m in at.markdown if "Subsystem Health" in m.value or "Active Registered Production Model" in m.value]
        self.assertGreater(len(headers), 0, "Expected health check or model governance headers on Settings page.")

        # Verify no plain secrets leaked in page markdown
        for m in at.markdown:
            self.assertNotIn("secret-test-token", m.value)
            self.assertNotIn("retention_pass", m.value)


if __name__ == "__main__":
    unittest.main()

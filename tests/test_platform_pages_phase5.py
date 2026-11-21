"""Integration Tests for Phase 5 Streamlit Pages via AppTest."""

from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

from dashboard.core.database import CampaignCreate, init_db
from dashboard.core.campaign_repository import create_campaign, delete_campaign, list_campaigns

BASE_DIR = Path(__file__).resolve().parents[1]
FAIRNESS_AUDIT_PATH = str(BASE_DIR / "dashboard" / "pages" / "fairness_audit.py")
CAMPAIGNS_PATH = str(BASE_DIR / "dashboard" / "pages" / "campaigns.py")


class TestPlatformPhase5Pages(unittest.TestCase):
    """Test suite verifying Streamlit page rendering and widget interactions for Phase 5."""

    def setUp(self):
        init_db()
        # Ensure at least one test campaign exists for rendering
        camps = list_campaigns()
        if not camps:
            create_campaign(
                CampaignCreate(
                    name="Default AppTest Campaign",
                    objective="AppTest verification campaign",
                    target_segments=["At-Risk"],
                    budget=5000.0,
                    allowed_actions=["fee_waiver", "loyalty_reward"],
                    start_date="2026-12-01",
                    end_date="2026-12-31",
                ),
                user="apptest",
            )

    def test_fairness_audit_page_renders_cleanly(self):
        """Verify fairness_audit.py renders without exceptions and contains legal disclaimer."""
        at = AppTest.from_file(FAIRNESS_AUDIT_PATH)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"fairness_audit.py raised: {at.exception}")

        # Verify legal compliance governance notice is displayed
        info_boxes = [m.value for m in at.info if "Model Governance & Legal Compliance Notice" in m.value]
        self.assertGreater(len(info_boxes), 0, "Expected legal compliance notice in fairness audit page.")

        # Verify disparity metrics are present
        dp_mentions = [m.value for m in at.markdown if "Demographic Parity" in m.value]
        self.assertGreater(len(dp_mentions), 0, "Expected Demographic Parity mention.")

    def test_campaigns_page_renders_cleanly(self):
        """Verify campaigns.py renders without exceptions and contains operational notice."""
        at = AppTest.from_file(CAMPAIGNS_PATH)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"campaigns.py raised: {at.exception}")

        # Verify operational notice ("No messages are sent to customers") is prominently displayed
        info_boxes = [m.value for m in at.info if "No messages are sent to customers" in m.value]
        self.assertGreater(len(info_boxes), 0, "Expected messaging disclaimer in campaigns page.")

        # Verify privacy notice ("CustomerId identifiers are persisted") is displayed
        privacy_boxes = [m.value for m in at.info if "CustomerId" in m.value and "Surname" in m.value]
        self.assertGreater(len(privacy_boxes), 0, "Expected privacy notice in campaigns page.")

    def test_campaigns_workspace_tabs_rendered(self):
        """Verify the 4 workspace tabs are present in campaigns page."""
        at = AppTest.from_file(CAMPAIGNS_PATH)
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        # Check tabs exist
        tabs = [getattr(t, "label", "") for t in at.tabs]
        self.assertTrue(any("Target Allocation" in t for t in tabs))
        self.assertTrue(any("Outcomes" in t for t in tabs))
        self.assertTrue(any("Results" in t for t in tabs))
        self.assertTrue(any("Audit Trail" in t for t in tabs))


if __name__ == "__main__":
    unittest.main()


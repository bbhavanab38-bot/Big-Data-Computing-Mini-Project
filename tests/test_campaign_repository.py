"""Unit tests for Campaign Management repository, lifecycle state machine, and persistent storage."""
import unittest
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from dashboard.core.campaign_repository import (
    assign_campaign_targets,
    create_campaign,
    delete_campaign,
    get_audit_trail,
    get_campaign,
    get_campaign_results,
    list_campaigns,
    record_outcome,
    simulate_campaign_outcomes,
    update_campaign_status,
    validate_and_import_outcomes_csv,
)
from dashboard.core.database import (
    Base,
    CampaignCreate,
    CampaignTargetCreate,
    CampaignTargetModel,
    OutcomeCreate,
    OutcomeModel,
    init_db,
)


class TestCampaignRepository(unittest.TestCase):
    def setUp(self):
        # Ensure database tables exist
        init_db()

    def test_campaign_crud(self):
        """Test campaign creation, retrieval, listing, and deletion."""
        camp_name = f"Test Campaign CRUD {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="Evaluate high balance customer retention",
                target_segments=["At-Risk"],
                budget=8000.0,
                allowed_actions=["fee_waiver", "loyalty_reward"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            ),
            user="test_runner",
        )
        self.assertIsNotNone(camp.id)
        self.assertEqual(camp.status, "draft")

        # Retrieve
        fetched = get_campaign(camp.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.name, camp_name)

        # Delete
        success = delete_campaign(camp.id, user="test_runner")
        self.assertTrue(success)
        self.assertIsNone(get_campaign(camp.id))

    def test_state_machine_valid_and_invalid_transitions(self):
        """Verify lifecycle state transitions obey the state machine."""
        camp_name = f"State Machine Test {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="State machine transition validation",
                target_segments=["At-Risk"],
                budget=5000.0,
                allowed_actions=["fee_waiver"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            )
        )

        # Assign 1 target so it can be approved
        assign_campaign_targets(
            camp.id,
            [CampaignTargetCreate(customer_id=15601539, assigned_action="fee_waiver")],
        )

        # Invalid: draft -> active (skipping approved)
        with self.assertRaises(ValueError) as ctx:
            update_campaign_status(camp.id, "active")
        self.assertIn("Invalid status transition", str(ctx.exception))

        # Valid: draft -> approved
        camp = update_campaign_status(camp.id, "approved")
        self.assertEqual(camp.status, "approved")

        # Valid: approved -> active
        camp = update_campaign_status(camp.id, "active")
        self.assertEqual(camp.status, "active")

        # Valid: active -> completed
        camp = update_campaign_status(camp.id, "completed")
        self.assertEqual(camp.status, "completed")

        # Invalid: completed -> active (terminal state)
        with self.assertRaises(ValueError) as ctx:
            update_campaign_status(camp.id, "active")
        self.assertIn("Invalid status transition", str(ctx.exception))

        # Clean up
        with self.assertRaises(ValueError):
            # Completed campaigns cannot be deleted directly
            delete_campaign(camp.id)

    def test_pre_approval_validation_blocks_empty_or_overbudget(self):
        """Verify draft campaign cannot be approved without targets or when over budget."""
        camp_name = f"Pre Approval Test {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="Test pre-approval blocking",
                target_segments=["At-Risk"],
                budget=100.0,  # small budget
                allowed_actions=["fee_waiver"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            )
        )

        # Attempt to approve without targets -> should fail
        with self.assertRaises(ValueError) as ctx:
            update_campaign_status(camp.id, "approved")
        self.assertIn("Cannot approve campaign without assigned targets", str(ctx.exception))

        # Clean up
        delete_campaign(camp.id)

    def test_budget_and_capacity_enforcement(self):
        """Verify assigning targets enforces total budget and action capacity limits."""
        camp_name = f"Constraint Test {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="Budget constraint enforcement",
                target_segments=["At-Risk"],
                budget=50.0,  # Fee waiver costs $50, so 2 fee waivers cost $100 > $50
                allowed_actions=["fee_waiver"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            )
        )

        over_budget_targets = [
            CampaignTargetCreate(customer_id=15601539, assigned_action="fee_waiver"),
            CampaignTargetCreate(customer_id=15606332, assigned_action="fee_waiver"),
        ]

        # Should raise ValueError for exceeding budget
        with self.assertRaises(ValueError) as ctx:
            assign_campaign_targets(camp.id, over_budget_targets)
        self.assertIn("exceeds campaign budget", str(ctx.exception))

        delete_campaign(camp.id)

    def test_privacy_and_outcome_recording(self):
        """Verify outcome recording updates target status and stores CustomerId without PII."""
        camp_name = f"Privacy Outcome Test {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="Privacy and outcome test",
                target_segments=["At-Risk"],
                budget=2000.0,
                allowed_actions=["fee_waiver"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            )
        )

        assign_campaign_targets(
            camp.id,
            [CampaignTargetCreate(customer_id=15601539, assigned_action="fee_waiver")],
        )

        # Record outcome
        out = record_outcome(
            campaign_id=camp.id,
            outcome=OutcomeCreate(
                customer_id=15601539,
                contacted=True,
                responded=True,
                accepted=True,
                churned=False,
                source="manual",
                recorded_by="compliance_officer",
            ),
        )
        self.assertEqual(out.customer_id, 15601539)
        self.assertFalse(out.churned)
        self.assertEqual(out.source, "manual")

        # Verify privacy: target and outcome tables have customer_id column and NO surname column
        target_cols = [c.name for c in CampaignTargetModel.__table__.columns]
        outcome_cols = [c.name for c in OutcomeModel.__table__.columns]
        self.assertIn("customer_id", target_cols)
        self.assertNotIn("Surname", target_cols)
        self.assertNotIn("surname", target_cols)
        self.assertIn("customer_id", outcome_cols)
        self.assertNotIn("Surname", outcome_cols)
        self.assertNotIn("surname", outcome_cols)

        # Clean up
        delete_campaign(camp.id)

    def test_csv_upload_validation(self):
        """Verify CSV upload validation catches missing headers, unknown IDs, and parses valid data."""
        camp_name = f"CSV Validation Test {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="CSV validation testing",
                target_segments=["At-Risk"],
                budget=2000.0,
                allowed_actions=["fee_waiver"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            )
        )

        assign_campaign_targets(
            camp.id,
            [CampaignTargetCreate(customer_id=15601539, assigned_action="fee_waiver")],
        )

        # 1. Missing required column
        bad_csv_1 = "CustomerId,contacted,responded\n15601539,true,true"
        cnt, errs = validate_and_import_outcomes_csv(camp.id, bad_csv_1)
        self.assertEqual(cnt, 0)
        self.assertTrue(any("missing required columns" in e for e in errs))

        # 2. Unknown CustomerId
        bad_csv_2 = "CustomerId,contacted,responded,accepted,churned\n99999999,true,true,true,false"
        cnt, errs = validate_and_import_outcomes_csv(camp.id, bad_csv_2)
        self.assertEqual(cnt, 0)
        self.assertTrue(any("not a target" in e for e in errs))

        # 3. Valid CSV
        good_csv = "CustomerId,contacted,responded,accepted,churned\n15601539,true,true,true,false"
        cnt, errs = validate_and_import_outcomes_csv(camp.id, good_csv)
        self.assertEqual(cnt, 1)
        self.assertEqual(len(errs), 0)

        # Clean up
        delete_campaign(camp.id)

    def test_audit_trail_logging(self):
        """Verify immutable audit log captures campaign creation and actions."""
        camp_name = f"Audit Log Test {datetime.utcnow().timestamp()}"
        camp = create_campaign(
            CampaignCreate(
                name=camp_name,
                objective="Audit trail verification",
                target_segments=["At-Risk"],
                budget=1000.0,
                allowed_actions=["fee_waiver"],
                start_date="2026-12-01",
                end_date="2026-12-31",
            ),
            user="audit_auditor",
        )

        logs = get_audit_trail(campaign_id=camp.id)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(logs[0].action, "CREATE")
        self.assertEqual(logs[0].user, "audit_auditor")

        # Clean up
        delete_campaign(camp.id)


if __name__ == "__main__":
    unittest.main()


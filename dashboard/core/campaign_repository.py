"""Campaign Management Repository with Transactional Lifecycle and Audit Trails.

Handles campaign CRUD, validated state machine transitions, budget/capacity constraint
enforcement, CSV outcome uploads, uplift simulation integration, and realized ROI calculation.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy.orm import Session

from dashboard.core.actions import load_action_catalog
from dashboard.core.clv import CLVParameters, compute_portfolio_clv
from dashboard.core.database import (
    AuditLogModel,
    CampaignCreate,
    CampaignModel,
    CampaignTargetCreate,
    CampaignTargetModel,
    OutcomeCreate,
    OutcomeModel,
    get_db_session,
    init_db,
)
from dashboard.core.uplift_simulator import simulate_ground_truth_outcomes
from src import config

# Permitted state machine transitions
ALLOWED_TRANSITIONS: Dict[str, set[str]] = {
    "draft": {"approved", "cancelled"},
    "approved": {"active", "cancelled"},
    "active": {"completed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


def _write_audit_log(
    session: Session,
    entity_type: str,
    entity_id: str,
    action: str,
    old_state: Optional[Dict[str, Any]] = None,
    new_state: Optional[Dict[str, Any]] = None,
    user: str = "operator",
) -> None:
    """Helper to record an immutable audit log entry within an active transaction."""
    log = AuditLogModel(
        entity_type=entity_type,
        entity_id=str(entity_id),
        action=action,
        old_state=json.dumps(old_state) if old_state else None,
        new_state=json.dumps(new_state) if new_state else None,
        user=user,
        timestamp=datetime.utcnow(),
    )
    session.add(log)


# =============================================================================
# Campaign CRUD
# =============================================================================

def create_campaign(data: CampaignCreate, user: str = "operator") -> CampaignModel:
    """Create a new campaign in 'draft' status after Pydantic schema validation."""
    init_db()
    with get_db_session() as session:
        # Check uniqueness of campaign name
        existing = session.query(CampaignModel).filter_by(name=data.name).first()
        if existing:
            raise ValueError(f"Campaign with name '{data.name}' already exists (ID: {existing.id}).")

        campaign = CampaignModel(
            name=data.name,
            objective=data.objective,
            target_segments=json.dumps(data.target_segments),
            budget=float(data.budget),
            allowed_actions=json.dumps(data.allowed_actions),
            start_date=data.start_date,
            end_date=data.end_date,
            status="draft",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        session.add(campaign)
        session.flush()

        _write_audit_log(
            session=session,
            entity_type="campaign",
            entity_id=str(campaign.id),
            action="CREATE",
            new_state={
                "name": campaign.name,
                "budget": campaign.budget,
                "status": "draft",
                "actions": data.allowed_actions,
            },
            user=user,
        )
        session.commit()
        session.refresh(campaign)
        return campaign


def get_campaign(campaign_id: int) -> Optional[CampaignModel]:
    """Retrieve campaign by ID."""
    with get_db_session() as session:
        return session.query(CampaignModel).filter_by(id=campaign_id).first()


def get_campaign_targets(campaign_id: int) -> List[CampaignTargetModel]:
    """Retrieve all assigned targets for a campaign."""
    with get_db_session() as session:
        return session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).all()


def get_campaign_outcomes(campaign_id: int) -> List[OutcomeModel]:
    """Retrieve all recorded outcomes for a campaign."""
    with get_db_session() as session:
        return session.query(OutcomeModel).filter_by(campaign_id=campaign_id).all()



def list_campaigns(status: Optional[str] = None) -> List[CampaignModel]:
    """List all campaigns, optionally filtered by status."""
    init_db()
    with get_db_session() as session:
        q = session.query(CampaignModel)
        if status:
            q = q.filter_by(status=status)
        return q.order_by(CampaignModel.created_at.desc()).all()


def delete_campaign(campaign_id: int, user: str = "operator", reason: str = "") -> bool:
    """Permanently delete a campaign. Allowed ONLY for 'draft' or 'cancelled' campaigns."""
    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign #{campaign_id} not found.")

        if campaign.status not in {"draft", "cancelled"}:
            raise ValueError(
                f"Cannot delete campaign #{campaign_id} in status '{campaign.status}'. "
                "Only 'draft' or 'cancelled' campaigns can be deleted."
            )

        snapshot = {
            "name": campaign.name,
            "status": campaign.status,
            "budget": campaign.budget,
            "reason": reason,
        }

        _write_audit_log(
            session=session,
            entity_type="campaign",
            entity_id=str(campaign_id),
            action="DELETE",
            old_state=snapshot,
            user=user,
        )
        session.delete(campaign)
        session.commit()
        return True


# =============================================================================
# State Machine Lifecycle Transitions
# =============================================================================

def update_campaign_status(
    campaign_id: int,
    new_status: str,
    user: str = "operator",
    reason: str = "",
) -> CampaignModel:
    """Execute a validated lifecycle state transition.
    
    Allowed transitions:
    - draft -> approved, cancelled
    - approved -> active, cancelled
    - active -> completed, cancelled
    - completed / cancelled -> terminal
    """
    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign #{campaign_id} not found.")

        current_status = campaign.status
        allowed = ALLOWED_TRANSITIONS.get(current_status, set())

        if new_status not in allowed:
            raise ValueError(
                f"Invalid status transition from '{current_status}' to '{new_status}'. "
                f"Permitted next states: {sorted(list(allowed)) or 'None (Terminal state)'}"
            )

        # Pre-approval validation: targets must exist, budget & capacity must be met
        if new_status == "approved":
            target_count = session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).count()
            if target_count == 0:
                raise ValueError("Cannot approve campaign without assigned targets. Assign targets first.")

            # Validate target costs against budget
            catalog = load_action_catalog()
            targets = session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).all()
            total_cost = sum(catalog.get_action(t.assigned_action).cost for t in targets)
            if total_cost > campaign.budget:
                raise ValueError(
                    f"Cannot approve campaign: Target cost (${total_cost:,.2f}) exceeds "
                    f"campaign budget (${campaign.budget:,.2f})."
                )

        old_state = {"status": current_status}
        new_state = {"status": new_status, "reason": reason}

        campaign.status = new_status
        campaign.updated_at = datetime.utcnow()

        action_name = "CANCEL" if new_status == "cancelled" else "STATUS_CHANGE"
        _write_audit_log(
            session=session,
            entity_type="campaign",
            entity_id=str(campaign_id),
            action=action_name,
            old_state=old_state,
            new_state=new_state,
            user=user,
        )

        session.commit()
        session.refresh(campaign)
        return campaign


# =============================================================================
# Target Assignment & Constraint Enforcement
# =============================================================================

def assign_campaign_targets(
    campaign_id: int,
    targets: List[CampaignTargetCreate],
    user: str = "operator",
) -> int:
    """Assign targets to a campaign with strict budget and capacity enforcement.
    
    Can only be performed on campaigns in 'draft' status.
    """
    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign #{campaign_id} not found.")

        if campaign.status != "draft":
            raise ValueError(
                f"Cannot modify targets for campaign #{campaign_id} in status '{campaign.status}'. "
                "Targets can only be assigned in 'draft' status."
            )

        allowed_actions = set(json.loads(campaign.allowed_actions))
        catalog = load_action_catalog()

        # Check action validity and calculate costs / capacities
        total_cost = 0.0
        action_counts: Dict[str, int] = {}
        unique_customers = set()

        for t in targets:
            if t.customer_id in unique_customers:
                raise ValueError(f"Duplicate customer ID {t.customer_id} in target payload.")
            unique_customers.add(t.customer_id)

            if t.assigned_action not in allowed_actions:
                raise ValueError(
                    f"Action '{t.assigned_action}' is not in allowed actions for this campaign: {sorted(list(allowed_actions))}"
                )

            action_cfg = catalog.get_action(t.assigned_action)
            total_cost += action_cfg.cost
            action_counts[t.assigned_action] = action_counts.get(t.assigned_action, 0) + 1

        # Check budget constraint
        if total_cost > campaign.budget:
            raise ValueError(
                f"Total allocation cost (${total_cost:,.2f}) exceeds campaign budget (${campaign.budget:,.2f})."
            )

        # Check capacity constraints
        for act_id, count in action_counts.items():
            act_cfg = catalog.get_action(act_id)
            if act_cfg.capacity is not None and count > act_cfg.capacity:
                raise ValueError(
                    f"Action '{act_cfg.name}' exceeds catalog capacity limit ({count} allocated > {act_cfg.capacity} max)."
                )

        # Clear existing targets for idempotency
        session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).delete()

        # Insert new targets (strictly storing customer_id only)
        for t in targets:
            model = CampaignTargetModel(
                campaign_id=campaign_id,
                customer_id=t.customer_id,
                assigned_action=t.assigned_action,
                status="pending",
                created_at=datetime.utcnow(),
            )
            session.add(model)

        campaign.updated_at = datetime.utcnow()
        _write_audit_log(
            session=session,
            entity_type="campaign",
            entity_id=str(campaign_id),
            action="ASSIGN_TARGETS",
            new_state={
                "target_count": len(targets),
                "total_cost": total_cost,
                "action_counts": action_counts,
            },
            user=user,
        )

        session.commit()
        return len(targets)


# =============================================================================
# Outcome Recording & CSV Validation
# =============================================================================

def record_outcome(campaign_id: int, outcome: OutcomeCreate, user: str = "operator") -> OutcomeModel:
    """Record an outcome for a single targeted customer."""
    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign #{campaign_id} not found.")

        # Customer must be in targets
        target = session.query(CampaignTargetModel).filter_by(
            campaign_id=campaign_id, customer_id=outcome.customer_id
        ).first()
        if not target:
            raise ValueError(f"Customer #{outcome.customer_id} is not an assigned target of campaign #{campaign_id}.")

        # Check existing outcome
        existing = session.query(OutcomeModel).filter_by(
            campaign_id=campaign_id, customer_id=outcome.customer_id
        ).first()

        if existing:
            existing.contacted = outcome.contacted
            existing.responded = outcome.responded
            existing.accepted = outcome.accepted
            existing.churned = outcome.churned
            existing.recorded_at = datetime.utcnow()
            existing.recorded_by = outcome.recorded_by
            existing.source = outcome.source
            model = existing
        else:
            model = OutcomeModel(
                campaign_id=campaign_id,
                customer_id=outcome.customer_id,
                contacted=outcome.contacted,
                responded=outcome.responded,
                accepted=outcome.accepted,
                churned=outcome.churned,
                recorded_at=datetime.utcnow(),
                recorded_by=outcome.recorded_by,
                source=outcome.source,
            )
            session.add(model)

        target.status = "completed"
        session.commit()
        session.refresh(model)
        return model


def validate_and_import_outcomes_csv(
    campaign_id: int,
    csv_text_or_bytes: str | bytes,
    user: str = "operator",
) -> Tuple[int, List[str]]:
    """Validate and batch import campaign outcomes from CSV.
    
    Required columns: CustomerId, contacted, responded, accepted, churned
    Returns:
        (imported_count, error_messages)
    """
    if isinstance(csv_text_or_bytes, bytes):
        text_data = csv_text_or_bytes.decode("utf-8", errors="replace")
    else:
        text_data = csv_text_or_bytes

    reader = csv.DictReader(io.StringIO(text_data))
    required_cols = {"CustomerId", "contacted", "responded", "accepted", "churned"}
    if not reader.fieldnames or not required_cols.issubset(set(reader.fieldnames)):
        missing = required_cols - set(reader.fieldnames or [])
        return 0, [f"CSV missing required columns: {sorted(list(missing))}"]

    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            return 0, [f"Campaign #{campaign_id} not found."]

        valid_targets = {
            t.customer_id for t in session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).all()
        }

        errors: List[str] = []
        valid_outcomes: List[OutcomeCreate] = []

        def _parse_bool(val: Any) -> bool:
            s = str(val).strip().lower()
            return s in {"true", "1", "yes", "y", "t"}

        for row_idx, row in enumerate(reader, start=2):
            try:
                cid = int(row["CustomerId"])
            except ValueError:
                errors.append(f"Row {row_idx}: Invalid CustomerId '{row.get('CustomerId')}'; must be an integer.")
                continue

            if cid not in valid_targets:
                errors.append(f"Row {row_idx}: CustomerId {cid} is not a target in campaign #{campaign_id}.")
                continue

            try:
                outcome = OutcomeCreate(
                    customer_id=cid,
                    contacted=_parse_bool(row["contacted"]),
                    responded=_parse_bool(row["responded"]),
                    accepted=_parse_bool(row["accepted"]),
                    churned=_parse_bool(row["churned"]),
                    source="uploaded",
                    recorded_by=user,
                )
                valid_outcomes.append(outcome)
            except Exception as e:
                errors.append(f"Row {row_idx}: Validation error: {str(e)}")

        if errors:
            return 0, errors

        # Upsert valid outcomes
        count = 0
        for out in valid_outcomes:
            existing = session.query(OutcomeModel).filter_by(
                campaign_id=campaign_id, customer_id=out.customer_id
            ).first()
            if existing:
                existing.contacted = out.contacted
                existing.responded = out.responded
                existing.accepted = out.accepted
                existing.churned = out.churned
                existing.recorded_at = datetime.utcnow()
                existing.recorded_by = user
                existing.source = "uploaded"
            else:
                session.add(
                    OutcomeModel(
                        campaign_id=campaign_id,
                        customer_id=out.customer_id,
                        contacted=out.contacted,
                        responded=out.responded,
                        accepted=out.accepted,
                        churned=out.churned,
                        recorded_at=datetime.utcnow(),
                        recorded_by=user,
                        source="uploaded",
                    )
                )
            count += 1

        _write_audit_log(
            session=session,
            entity_type="outcome",
            entity_id=str(campaign_id),
            action="CSV_UPLOAD",
            new_state={"count": count, "source": "uploaded"},
            user=user,
        )
        session.commit()
        return count, []


# =============================================================================
# Outcome Simulator Integration (Clearly Tagged)
# =============================================================================

def simulate_campaign_outcomes(campaign_id: int, user: str = "simulator") -> int:
    """Generate simulated customer outcomes using the uplift simulator's response function.
    
    CRITICAL TRANSPARENCY RULE: All outcomes generated here are tagged `source='simulated'`.
    They cannot be mistaken for real-world interactions.
    """
    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign #{campaign_id} not found.")

        targets = session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).all()
        if not targets:
            raise ValueError("No targets assigned to this campaign to simulate.")

        target_cids = [t.customer_id for t in targets]
        action_map = {t.customer_id: t.assigned_action for t in targets}

        # Load customer features for these targets
        df = pd.read_csv(config.SCORED_DATASET_FILE)
        cust_sub = df[df["CustomerId"].isin(target_cids)].copy()

        # Run uplift simulator ground truth response
        sim_results = simulate_ground_truth_outcomes(cust_sub, action_map)

        sim_count = 0
        for _, row in sim_results.iterrows():
            cid = int(row["CustomerId"])
            churned = bool(row["simulated_churn"])
            responded = bool(row["simulated_response"])
            accepted = bool(row.get("simulated_accepted", responded))

            existing = session.query(OutcomeModel).filter_by(
                campaign_id=campaign_id, customer_id=cid
            ).first()

            if existing:
                existing.contacted = True
                existing.responded = responded
                existing.accepted = accepted
                existing.churned = churned
                existing.recorded_at = datetime.utcnow()
                existing.recorded_by = user
                existing.source = "simulated"
            else:
                session.add(
                    OutcomeModel(
                        campaign_id=campaign_id,
                        customer_id=cid,
                        contacted=True,
                        responded=responded,
                        accepted=accepted,
                        churned=churned,
                        recorded_at=datetime.utcnow(),
                        recorded_by=user,
                        source="simulated",
                    )
                )
            sim_count += 1

        _write_audit_log(
            session=session,
            entity_type="outcome",
            entity_id=str(campaign_id),
            action="SIMULATE_OUTCOMES",
            new_state={"simulated_count": sim_count, "source": "simulated"},
            user=user,
        )
        session.commit()
        return sim_count


# =============================================================================
# Campaign Results & Reporting
# =============================================================================

def get_campaign_results(campaign_id: int) -> Dict[str, Any]:
    """Calculate campaign summary statistics, spend, and realized vs planned net benefit."""
    with get_db_session() as session:
        campaign = session.query(CampaignModel).filter_by(id=campaign_id).first()
        if not campaign:
            raise ValueError(f"Campaign #{campaign_id} not found.")

        targets = session.query(CampaignTargetModel).filter_by(campaign_id=campaign_id).all()
        outcomes = session.query(OutcomeModel).filter_by(campaign_id=campaign_id).all()

        target_count = len(targets)
        catalog = load_action_catalog()

        # Planned spend
        planned_spend = sum(catalog.get_action(t.assigned_action).cost for t in targets)

        outcome_by_cid = {o.customer_id: o for o in outcomes}
        action_by_cid = {t.customer_id: t.assigned_action for t in targets}

        contacted_count = sum(1 for o in outcomes if o.contacted)
        responded_count = sum(1 for o in outcomes if o.responded)
        accepted_count = sum(1 for o in outcomes if o.accepted)
        churned_count = sum(1 for o in outcomes if o.churned)
        retained_count = sum(1 for o in outcomes if not o.churned)

        response_rate = float(responded_count / contacted_count) if contacted_count > 0 else 0.0
        retention_rate = float(retained_count / len(outcomes)) if len(outcomes) > 0 else 0.0
        churn_rate = float(churned_count / len(outcomes)) if len(outcomes) > 0 else 0.0

        # Spend incurred (only for contacted customers)
        actual_spend = sum(
            catalog.get_action(action_by_cid[o.customer_id]).cost
            for o in outcomes
            if o.contacted and o.customer_id in action_by_cid
        )

        # Realized net benefit: Retained CLV minus spend
        df = pd.read_csv(config.SCORED_DATASET_FILE)
        retained_cids = [o.customer_id for o in outcomes if not o.churned]
        retained_df = df[df["CustomerId"].isin(retained_cids)]

        clv_params = CLVParameters()
        if len(retained_df) > 0:
            retained_clvs = compute_portfolio_clv(retained_df, clv_params)
            total_retained_clv = float(retained_clvs.sum())
        else:
            total_retained_clv = 0.0

        realised_net_benefit = total_retained_clv - actual_spend
        has_simulated_outcomes = any(o.source == "simulated" for o in outcomes)

        # Per-action breakdown
        action_stats: Dict[str, Dict[str, Any]] = {}
        for act_id in json.loads(campaign.allowed_actions):
            act_cfg = catalog.get_action(act_id)
            act_targets = [t for t in targets if t.assigned_action == act_id]
            act_cids = {t.customer_id for t in act_targets}
            act_outcomes = [o for o in outcomes if o.customer_id in act_cids]

            act_contacted = sum(1 for o in act_outcomes if o.contacted)
            act_responded = sum(1 for o in act_outcomes if o.responded)
            act_retained = sum(1 for o in act_outcomes if not o.churned)

            action_stats[act_id] = {
                "name": act_cfg.name,
                "cost": act_cfg.cost,
                "targets": len(act_targets),
                "contacted": act_contacted,
                "responded": act_responded,
                "retained": act_retained,
                "response_rate": float(act_responded / act_contacted) if act_contacted > 0 else 0.0,
                "retention_rate": float(act_retained / len(act_outcomes)) if len(act_outcomes) > 0 else 0.0,
                "total_spend": act_contacted * act_cfg.cost,
            }

        return {
            "campaign_id": campaign.id,
            "campaign_name": campaign.name,
            "status": campaign.status,
            "budget": campaign.budget,
            "target_count": target_count,
            "outcome_count": len(outcomes),
            "contacted_count": contacted_count,
            "responded_count": responded_count,
            "response_rate": response_rate,
            "accepted_count": accepted_count,
            "retained_count": retained_count,
            "retention_rate": retention_rate,
            "churned_count": churned_count,
            "churn_rate": churn_rate,
            "planned_spend": planned_spend,
            "actual_spend": actual_spend,
            "budget_utilization": float(actual_spend / campaign.budget) if campaign.budget > 0 else 0.0,
            "total_retained_clv": total_retained_clv,
            "realised_net_benefit": realised_net_benefit,
            "has_simulated_outcomes": has_simulated_outcomes,
            "per_action_stats": action_stats,
        }


def get_audit_trail(campaign_id: Optional[int] = None, limit: int = 50) -> List[AuditLogModel]:
    """Retrieve audit trail logs chronologically."""
    with get_db_session() as session:
        q = session.query(AuditLogModel)
        if campaign_id is not None:
            q = q.filter_by(entity_id=str(campaign_id))
        return q.order_by(AuditLogModel.timestamp.desc()).limit(limit).all()

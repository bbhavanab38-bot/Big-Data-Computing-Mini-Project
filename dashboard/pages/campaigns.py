"""Campaign Management Page with Persistent Storage, Lifecycle State Machine, and Audit Trails.

Enforces budget and capacity constraints, validates lifecycle transitions, manages customer targets,
provides CSV outcome imports, integrates uplift simulation with explicit simulation tagging,
and calculates realized financial net benefit.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
from datetime import datetime
import streamlit as st
import pandas as pd

from dashboard.core.actions import load_action_catalog
from dashboard.core.campaign_repository import (
    assign_campaign_targets,
    create_campaign,
    delete_campaign,
    get_audit_trail,
    get_campaign,
    get_campaign_targets,
    get_campaign_results,
    list_campaigns,
    record_outcome,
    simulate_campaign_outcomes,
    update_campaign_status,
    validate_and_import_outcomes_csv,
)
from dashboard.core.database import (
    CampaignCreate,
    CampaignTargetCreate,
    OutcomeCreate,
    init_db,
)
from dashboard.core.optimization import solve_retention_allocation
from dashboard.core.platform_data import load_scored_dataset
from dashboard.ui.charts import build_campaign_outcomes_bar
from dashboard.ui.components import (
    apply_page_styles,
    empty_state,
    insight_banner,
    kpi_card,
    page_header,
    section_header,
)


def render():
    apply_page_styles()
    init_db()

    page_header(
        title="Campaign Management & Lifecycle Execution",
        description="Persistent campaign portfolio tracking, target allocation, lifecycle governance, CSV outcome uploads, and realized ROI auditing.",
        badges=["SQLITE / SQLALCHEMY 2.0", "PERSISTENT DB", "AUDIT TRAIL", "GOVERNANCE"],
    )

    # Operational Disclaimers
    st.info(
        "📣 **Operational Notice**: No messages are sent to customers. This application has no messaging integration.\n\n"
        "🔒 **Privacy Notice**: Customer privacy is protected. Only anonymous `CustomerId` identifiers are persisted in "
        "the campaign database. Personal fields such as `Surname` are never stored."
    )

    catalog = load_action_catalog()
    all_campaigns = list_campaigns()

    # Top Control Bar
    top_c1, top_c2 = st.columns([3, 1])
    with top_c1:
        st.caption(f"Active Campaign Database: `{len(all_campaigns)}` total campaigns on file.")
    with top_c2:
        show_create_form = st.toggle("➕ New Campaign", value=False)

    # Create Campaign Form
    if show_create_form:
        st.markdown("---")
        section_header("Create Retention Campaign", "Define campaign parameters, budget, and allowed retention interventions.")
        
        with st.form("create_campaign_form", clear_on_submit=False):
            fc1, fc2 = st.columns(2)
            with fc1:
                camp_name = st.text_input("Campaign Name *", placeholder="e.g. Q4 High-Balance Churn Defense")
                camp_obj = st.text_area("Objective *", placeholder="e.g. Protect at-risk accounts with balance > $50k from competitor attrition")
                camp_budget = st.number_input("Campaign Budget ($) *", min_value=100.0, max_value=500000.0, value=10000.0, step=500.0)

            with fc2:
                # Segment choices
                df = load_scored_dataset()
                segment_opts = sorted(list(df["SegmentName"].dropna().unique())) if not df.empty else ["At-Risk", "Digital Affluents"]
                camp_segments = st.multiselect("Target Customer Segments *", options=segment_opts, default=segment_opts[:2])
                
                # Action choices
                action_opts = [a.id for a in catalog.actions if a.id != "no_intervention"]
                action_labels = {a.id: f"{a.name} (${a.cost:,.0f})" for a in catalog.actions}
                camp_actions = st.multiselect(
                    "Allowed Retention Actions *",
                    options=action_opts,
                    format_func=lambda x: action_labels.get(x, x),
                    default=action_opts[:3],
                )

                d1, d2 = st.columns(2)
                with d1:
                    start_d = st.date_input("Start Date (Admin Metadata)", value=datetime.utcnow().date())
                with d2:
                    end_d = st.date_input("End Date (Admin Metadata)", value=datetime.utcnow().date())

            submit_create = st.form_submit_button("Create Campaign (Draft)", type="primary", width="stretch")

            if submit_create:
                if not camp_name or len(camp_name.strip()) < 3:
                    st.error("Campaign name must be at least 3 characters.")
                elif not camp_obj or len(camp_obj.strip()) < 5:
                    st.error("Objective must be at least 5 characters.")
                elif not camp_segments:
                    st.error("Please select at least one target segment.")
                elif not camp_actions:
                    st.error("Please select at least one allowed retention action.")
                elif end_d < start_d:
                    st.error("End date cannot be earlier than start date.")
                else:
                    try:
                        new_c = create_campaign(
                            CampaignCreate(
                                name=camp_name.strip(),
                                objective=camp_obj.strip(),
                                target_segments=camp_segments,
                                budget=float(camp_budget),
                                allowed_actions=camp_actions,
                                start_date=start_d.strftime("%Y-%m-%d"),
                                end_date=end_d.strftime("%Y-%m-%d"),
                            ),
                            user="operator",
                        )
                        st.success(f"Campaign '{new_c.name}' created successfully in 'draft' status (ID: #{new_c.id})!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create campaign: {str(e)}")

    if not all_campaigns:
        empty_state(
            title="No Campaigns Found",
            message="No campaigns currently exist in the database. Toggle '➕ New Campaign' above to create your first retention program.",
            icon="📋",
        )
        return

    # Campaign Selector & Status Overview
    st.markdown("---")
    sc1, sc2 = st.columns([2.5, 1.5])
    
    with sc1:
        campaign_map = {f"#{c.id}: {c.name} [{c.status.upper()}]": c.id for c in all_campaigns}
        selected_label = st.selectbox("Select Active Campaign Workspace", options=list(campaign_map.keys()), index=0)
        selected_campaign_id = campaign_map[selected_label]
        campaign = get_campaign(selected_campaign_id)

    with sc2:
        if campaign:
            st.markdown(f"**Status**: `{campaign.status.upper()}` | **Budget**: `${campaign.budget:,.2f}`")
            st.caption(f"Created: {campaign.created_at.strftime('%Y-%m-%d %H:%M UTC')}")

    if not campaign:
        st.error("Selected campaign not found.")
        return

    # Lifecycle State Machine & Actions
    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2, b_col3, b_col4, b_col5 = st.columns(5)

    with b_col1:
        if campaign.status == "draft":
            if st.button("✅ Approve Campaign", type="primary", width="stretch"):
                try:
                    update_campaign_status(campaign.id, "approved", user="operator")
                    st.success("Campaign approved! Targets and budget verified.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Approval failed: {str(e)}")
        else:
            st.button("✅ Approve", disabled=True, width="stretch")

    with b_col2:
        if campaign.status == "approved":
            if st.button("🚀 Launch (Activate)", type="primary", width="stretch"):
                try:
                    update_campaign_status(campaign.id, "active", user="operator")
                    st.success("Campaign is now ACTIVE.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Activation failed: {str(e)}")
        else:
            st.button("🚀 Launch", disabled=True, width="stretch")

    with b_col3:
        if campaign.status == "active":
            if st.button("🏁 Mark Completed", width="stretch"):
                try:
                    update_campaign_status(campaign.id, "completed", user="operator")
                    st.success("Campaign marked completed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Completion failed: {str(e)}")
        else:
            st.button("🏁 Complete", disabled=True, width="stretch")

    with b_col4:
        # Cancel Flow
        if campaign.status in {"draft", "approved", "active"}:
            with st.popover("⛔ Cancel Campaign", width="stretch"):
                st.markdown("**Cancel Campaign Confirmation**")
                cancel_reason = st.text_input("Cancellation Reason *", placeholder="e.g. Budget reallocated or superseded")
                confirm_cancel = st.checkbox("I confirm cancellation of this campaign.")
                if st.button("Confirm Cancel", type="primary"):
                    if not cancel_reason:
                        st.error("Reason is required.")
                    elif not confirm_cancel:
                        st.error("Please check the confirmation box.")
                    else:
                        try:
                            update_campaign_status(campaign.id, "cancelled", user="operator", reason=cancel_reason)
                            st.warning("Campaign cancelled and logged to audit trail.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Cancellation failed: {str(e)}")
        else:
            st.button("⛔ Cancel", disabled=True, width="stretch")

    with b_col5:
        # Delete Flow (Draft or Cancelled only)
        if campaign.status in {"draft", "cancelled"}:
            with st.popover("🗑️ Delete Campaign", width="stretch"):
                st.markdown(f"**Delete Campaign #{campaign.id}**")
                st.caption("Permanent deletion. This cannot be undone.")
                confirm_del = st.checkbox("Confirm permanent deletion.")
                if st.button("Permanently Delete", type="primary"):
                    if confirm_del:
                        try:
                            delete_campaign(campaign.id, user="operator", reason="Operator requested deletion")
                            st.success("Campaign deleted.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {str(e)}")
                    else:
                        st.error("Please confirm deletion.")
        else:
            st.button("🗑️ Delete", disabled=True, width="stretch")

    # Main Campaign Workspace Tabs
    st.markdown("<br>", unsafe_allow_html=True)
    tab_targets, tab_outcomes, tab_results, tab_audit = st.tabs([
        "🎯 Target Allocation",
        "📥 Outcomes & Simulation",
        "📊 Results & Realized Net Benefit",
        "📜 Audit Trail",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: TARGET ALLOCATION
    # -------------------------------------------------------------------------
    with tab_targets:
        section_header("Customer Target Allocation", f"Assign customers to interventions within the ${campaign.budget:,.2f} budget.")

        if campaign.status != "draft":
            st.info(f"Targets are locked because campaign is in `{campaign.status.upper()}` status. Targets can only be modified in `DRAFT` status.")

        t_col1, t_col2 = st.columns([1.5, 1])

        with t_col1:
            st.markdown("#### Option A: Prescriptive Optimization Import")
            st.caption("Import the Decision Engine's MILP-optimized customer assignment maximizing Net Benefit subject to budget and capacity.")

            st.warning("⚠️ **Simulated Uplift Notice**: The prescriptive optimizer utilizes synthetic uplift response assumptions, not real-world pilot data.")

            import_opt_clicked = st.button(
                "⚡ Import Optimized Allocations from Decision Engine",
                type="primary",
                disabled=(campaign.status != "draft"),
            )

            if import_opt_clicked:
                with st.spinner("Running MILP optimization and compiling customer targets..."):
                    df = load_scored_dataset()
                    allowed_acts = json.loads(campaign.allowed_actions)
                    
                    # Run allocation
                    alloc_res = solve_retention_allocation(
                        customers_df=df,
                        budget=campaign.budget,
                        allowed_action_ids=allowed_acts,
                        catalog=catalog,
                    )

                    new_targets = [
                        CampaignTargetCreate(customer_id=a["customer_id"], assigned_action=a["assigned_action"])
                        for a in alloc_res.assignments
                    ]

                    try:
                        count = assign_campaign_targets(campaign.id, new_targets, user="optimizer_import")
                        st.success(f"Successfully imported {count:,} customer targets! Total planned cost: ${alloc_res.total_cost:,.2f}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Allocation import failed: {str(e)}")

        with t_col2:
            st.markdown("#### Option B: Target Capacity & Constraints")
            # Display current targets count and spend
            res = get_campaign_results(campaign.id)
            kpi_card(
                label="Targeted Accounts",
                value=f"{res['target_count']:,}",
                delta=f"${res['planned_spend']:,.2f} Planned",
                delta_basis=f"Budget: ${campaign.budget:,.2f}",
                delta_direction="normal" if res['planned_spend'] <= campaign.budget else "inverse",
            )

        # Show target assignments table
        with st.expander("View Assigned Targets Roster", expanded=True):
            targets = get_campaign_targets(campaign.id)
            if targets:
                t_rows = [
                    {
                        "CustomerId": t.customer_id,
                        "Assigned Action": catalog.get_action(t.assigned_action).name,
                        "Cost ($)": f"${catalog.get_action(t.assigned_action).cost:,.2f}",
                        "Status": t.status,
                        "Assigned At": t.created_at.strftime("%Y-%m-%d %H:%M"),
                    }
                    for t in targets[:500]
                ]
                st.dataframe(pd.DataFrame(t_rows), width="stretch", hide_index=True)
                if len(targets) > 500:
                    st.caption(f"Showing first 500 of {len(targets):,} assigned targets.")
            else:
                st.info("No targets currently assigned to this campaign.")

    # -------------------------------------------------------------------------
    # TAB 2: OUTCOMES & SIMULATION
    # -------------------------------------------------------------------------
    with tab_outcomes:
        section_header("Record Campaign Outcomes", "Capture response and retention outcomes through CSV upload, manual entry, or uplift simulation.")

        out_c1, out_c2 = st.columns(2)

        with out_c1:
            st.markdown("#### Mode 1: Batch Outcomes CSV Upload")
            st.caption("Upload realized customer response and churn outcomes. Schema is strictly validated.")

            csv_file = st.file_uploader("Upload Outcomes CSV", type=["csv"], help="Headers: CustomerId, contacted, responded, accepted, churned")
            if csv_file:
                if st.button("Validate and Ingest CSV", type="primary"):
                    with st.spinner("Validating CSV rows against campaign targets..."):
                        content = csv_file.getvalue()
                        cnt, errs = validate_and_import_outcomes_csv(campaign.id, content, user="csv_upload")
                        if errs:
                            st.error(f"CSV Validation Failed with {len(errs)} error(s):")
                            for err in errs[:5]:
                                st.write(f"- {err}")
                            if len(errs) > 5:
                                st.write(f"... and {len(errs)-5} more errors.")
                        else:
                            st.success(f"Successfully ingested {cnt:,} outcomes from CSV with source='uploaded'!")
                            st.rerun()

            with st.expander("CSV Template & Specification"):
                st.code(
                    "CustomerId,contacted,responded,accepted,churned\n"
                    "15601539,true,true,true,false\n"
                    "15606332,true,false,false,true",
                    language="csv"
                )

        with out_c2:
            st.markdown("#### Mode 2: Simulated Response Generator")
            insight_banner(
                title="SIMULATED PILOT RESPONSE ENGINE",
                text="Generates customer responses and churn outcomes using the synthetic uplift simulator's response function. "
                     "All generated rows are explicitly tagged with source='simulated' so they can never be confused with real data.",
                icon="🧪",
                badge="SIMULATOR",
            )

            sim_btn = st.button("⚡ Simulate Customer Responses", type="primary", width="stretch")
            if sim_btn:
                try:
                    sim_count = simulate_campaign_outcomes(campaign.id, user="simulator")
                    st.success(f"Generated {sim_count:,} simulated customer outcomes! All tagged source='simulated'.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")

        # Manual single customer entry
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("Mode 3: Manual Single Customer Outcome Entry"):
            with st.form("manual_outcome_form"):
                mo_c1, mo_c2, mo_c3 = st.columns(3)
                with mo_c1:
                    mo_cid = st.number_input("Customer ID", min_value=1, step=1)
                with mo_c2:
                    mo_contacted = st.checkbox("Customer Contacted", value=True)
                    mo_responded = st.checkbox("Customer Responded", value=True)
                with mo_c3:
                    mo_accepted = st.checkbox("Offer Accepted", value=True)
                    mo_churned = st.checkbox("Customer Churned (Exited)", value=False)

                mo_submit = st.form_submit_button("Record Manual Outcome")
                if mo_submit:
                    try:
                        record_outcome(
                            campaign_id=campaign.id,
                            outcome=OutcomeCreate(
                                customer_id=int(mo_cid),
                                contacted=mo_contacted,
                                responded=mo_responded,
                                accepted=mo_accepted,
                                churned=mo_churned,
                                source="manual",
                                recorded_by="manual_entry",
                            ),
                            user="manual_entry",
                        )
                        st.success(f"Recorded manual outcome for Customer #{mo_cid}!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to record outcome: {str(e)}")

    # -------------------------------------------------------------------------
    # TAB 3: CAMPAIGN RESULTS & REALIZED ROI
    # -------------------------------------------------------------------------
    with tab_results:
        section_header("Realized Campaign Results & Financial Impact", "Actual response rates, retention lift, spend incurred, and realized Net Benefit.")

        res = get_campaign_results(campaign.id)

        if res["has_simulated_outcomes"]:
            st.warning("⚠️ **Simulated Outcomes Notice**: Results include outcomes generated by the uplift simulator (`source='simulated'`). These are synthetic test figures.")

        # KPI Cards
        rk1, rk2, rk3, rk4, rk5 = st.columns(5)

        with rk1:
            kpi_card(
                label="Targeted / Contacted",
                value=f"{res['contacted_count']:,}",
                delta=f"{res['target_count']:,} Allocated",
                delta_basis=f"{float(res['contacted_count']/res['target_count'])*100:.1f}% outreach rate" if res['target_count'] > 0 else "0%",
            )

        with rk2:
            kpi_card(
                label="Response Rate",
                value=f"{res['response_rate']*100:.1f}%",
                delta=f"{res['responded_count']:,} Engaged",
                delta_basis="Contacted accounts responding to outreach",
            )

        with rk3:
            kpi_card(
                label="Retention Rate",
                value=f"{res['retention_rate']*100:.1f}%",
                delta=f"{res['retained_count']:,} Accounts Saved",
                delta_basis=f"Churn rate: {res['churn_rate']*100:.1f}%",
                delta_direction="normal" if res['retention_rate'] >= 0.75 else "inverse",
            )

        with rk4:
            kpi_card(
                label="Spend / Budget",
                value=f"${res['actual_spend']:,.2f}",
                delta=f"${res['budget']:,.2f} Cap",
                delta_basis=f"{res['budget_utilization']*100:.1f}% budget utilized",
                delta_direction="normal" if res['actual_spend'] <= res['budget'] else "inverse",
            )

        with rk5:
            kpi_card(
                label="Realized Net Benefit",
                value=f"${res['realised_net_benefit']:,.2f}",
                delta=f"${res['total_retained_clv']:,.2f} Retained CLV",
                delta_basis="Total saved customer value minus actual campaign spend",
                delta_direction="normal" if res['realised_net_benefit'] > 0 else "inverse",
            )

        # Per-action breakdown table & Funnel Chart
        st.markdown("<br>", unsafe_allow_html=True)
        act_stats = res["per_action_stats"]
        if act_stats:
            p_actions = []
            p_targets = []
            p_contacted = []
            p_responded = []
            p_retained = []

            rows = []
            for act_id, s in act_stats.items():
                p_actions.append(s["name"])
                p_targets.append(s["targets"])
                p_contacted.append(s["contacted"])
                p_responded.append(s["responded"])
                p_retained.append(s["retained"])

                rows.append({
                    "Intervention": s["name"],
                    "Unit Cost": f"${s['cost']:,.2f}",
                    "Targets": f"{s['targets']:,}",
                    "Contacted": f"{s['contacted']:,}",
                    "Responded": f"{s['responded']:,}",
                    "Response Rate": f"{s['response_rate']*100:.1f}%",
                    "Retained": f"{s['retained']:,}",
                    "Retention Rate": f"{s['retention_rate']*100:.1f}%",
                    "Spend": f"${s['total_spend']:,.2f}",
                })

            col_ct, col_cf = st.columns([1.2, 1.3])
            with col_ct:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            with col_cf:
                fig_act = build_campaign_outcomes_bar(
                    action_names=p_actions,
                    targets=p_targets,
                    contacted=p_contacted,
                    responded=p_responded,
                    retained=p_retained,
                    title="Intervention Progression Funnel",
                )
                st.plotly_chart(fig_act, width="stretch")

    # -------------------------------------------------------------------------
    # TAB 4: AUDIT TRAIL
    # -------------------------------------------------------------------------
    with tab_audit:
        section_header("Campaign Audit Trail", f"Immutable chronological event log for Campaign #{campaign.id}.")

        audit_logs = get_audit_trail(campaign_id=campaign.id, limit=50)

        if audit_logs:
            log_rows = [
                {
                    "Timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "Action": entry.action,
                    "Entity": entry.entity_type,
                    "User": entry.user,
                    "Old State": entry.old_state or "-",
                    "New State": entry.new_state or "-",
                }
                for entry in audit_logs
            ]
            st.dataframe(pd.DataFrame(log_rows), width="stretch", hide_index=True)
        else:
            st.info("No audit logs recorded for this campaign.")


if __name__ == "__main__":
    render()


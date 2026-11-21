"""Retention Decision Engine Page.

Prescriptive decision optimization allocating retention interventions subject to
total campaign budget, action capacity constraints, and customer-level CLV proxies.
Includes causal uplift modeling (T-Learner), policy comparison benchmarks,
and CSV export.

Pure Streamlit presentation calling pure-Python modules in dashboard/core/.
"""
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np

from dashboard.ui.theme import (
    CARD_BG,
    SECONDARY_ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from dashboard.ui.components import (
    apply_page_styles,
    insight_banner,
    kpi_card,
    page_header,
    section_header,
    render_html,
)
from dashboard.ui.charts import (
    build_policy_comparison_bar,
)
from dashboard.core.platform_data import load_scored_dataset
from dashboard.core.actions import load_action_catalog
from dashboard.core.clv import CLVParameters, compute_portfolio_clv
from dashboard.core.uplift_simulator import (
    TLearnerUplift,
    run_randomised_pilot,
)
from dashboard.core.optimization import (
    compare_policies,
    solve_retention_allocation,
)


def render_decision_engine():
    """Render the Retention Decision Engine interface."""
    apply_page_styles()
    page_header(
        title="Retention Decision Engine",
        description="Prescriptive optimization and causal intervention allocation under operational constraints.",
        badges=["SYNTHETIC EXPERIMENT"],
    )

    # Mandatory Simulation Disclaimer Banner
    insight_banner(
        title="Causal Simulation Protocol",
        text=(
            "Simulated uplift: not real-world measured. The underlying bank dataset contains no historical "
            "treatment/control intervention records. Uplift and intervention response curves are generated "
            "via a rigorous heterogeneous simulation protocol and learned with scikit-learn T-Learners."
        ),
        icon="⚠️",
        badge="SIMULATED UPLIFT",
    )

    # Load data and catalog
    df_scored = load_scored_dataset()
    catalog = load_action_catalog()
    non_control_actions = catalog.non_control_actions()

    # -------------------------------------------------------------------------
    # SIDEBAR: Optimization Controls & Economic Assumptions
    # -------------------------------------------------------------------------
    render_html(
        """
        <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 0.25rem;">
            CAMPAIGN CONSTRAINTS
        </div>
        """,
        container=st.sidebar,
    )

    budget_input = st.sidebar.number_input(
        "Total Budget (€)",
        min_value=500.0,
        max_value=200000.0,
        value=15000.0,
        step=1000.0,
        key="de_budget_input",
    )

    with st.sidebar.expander("Action Capacities (Operational Limits)", expanded=False):
        cap_overrides = {}
        for act in non_control_actions:
            default_cap = act.capacity or 500
            cap_val = st.number_input(
                f"{act.name} Max Capacity",
                min_value=10,
                max_value=5000,
                value=int(default_cap),
                step=50,
                key=f"de_cap_{act.id}",
            )
            cap_overrides[act.id] = cap_val

    with st.sidebar.expander("CLV Valuation Assumptions", expanded=False):
        st.caption("Assumption-based discounted cash flow proxy parameters:")
        nim_val = st.slider("Net Interest Margin (NIM %)", 1.0, 5.0, 2.5, 0.1, key="de_nim") / 100.0
        fee_val = st.slider("Annual Fee per Product (€)", 10.0, 100.0, 40.0, 5.0, key="de_fee")
        serv_val = st.slider("Annual Servicing Cost (€)", 10.0, 100.0, 35.0, 5.0, key="de_serv")
        horizon_val = st.slider("Horizon Window (Years)", 1, 5, 3, 1, key="de_horiz")
        disc_val = st.slider("Discount Rate (%)", 3.0, 15.0, 8.0, 0.5, key="de_disc") / 100.0

        clv_params = CLVParameters(
            nim=nim_val,
            fee_per_product=fee_val,
            servicing_cost=serv_val,
            horizon_years=horizon_val,
            discount_rate=disc_val,
            min_clv_floor=50.0,
        )

    # -------------------------------------------------------------------------
    # CACHE & RUN OPTIMIZATION
    # -------------------------------------------------------------------------
    col_run, col_status = st.columns([1.8, 3.2])
    with col_run:
        run_btn = st.button("Run Optimization Engine", type="primary", width="stretch")

    # Check session state for cached optimization run
    if run_btn or "last_optim_result" not in st.session_state:
        with st.spinner("Simulating pilot campaign, training T-Learners, and solving MILP allocation..."):
            # Step 1: Compute CLV
            df_scored["EstimatedCLV"] = compute_portfolio_clv(df_scored, params=clv_params)

            # Step 2: Run simulated pilot
            pilot = run_randomised_pilot(df_scored, catalog=catalog, seed=42)
            
            # Step 3: Train T-Learner on empirical pilot observations
            t_learner = TLearnerUplift()
            t_learner.fit(
                pilot.data[pilot.feature_cols],
                pilot.data["AssignedAction"],
                pilot.data["ObservedChurn"],
            )

            # Step 4: Predict multi-action uplifts on full portfolio
            pred_uplifts = t_learner.predict_all_uplifts(df_scored[pilot.feature_cols])

            # Step 5: Solve MILP
            optim_res = solve_retention_allocation(
                customer_ids=df_scored["CustomerId"].tolist(),
                clv_values=df_scored["EstimatedCLV"].values,
                uplift_matrix=pred_uplifts,
                catalog=catalog,
                budget=float(budget_input),
                per_action_capacities=cap_overrides,
                method="milp",
            )

            # Step 6: Policy benchmark comparison
            policy_cmp = compare_policies(df_scored, optim_res, catalog=catalog)

            # Store in session state
            st.session_state["last_optim_result"] = optim_res
            st.session_state["last_policy_cmp"] = policy_cmp
            st.session_state["last_t_learner"] = t_learner
            st.session_state["last_pred_uplifts"] = pred_uplifts
            # Store customer action mapping for Customer Intelligence lookup
            st.session_state["recommended_actions_map"] = dict(
                zip(optim_res.assignments["CustomerId"], optim_res.assignments["AssignedAction"])
            )

    optim_res = st.session_state["last_optim_result"]
    policy_cmp = st.session_state["last_policy_cmp"]

    # -------------------------------------------------------------------------
    # KPI METRIC CARDS
    # -------------------------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    total_contacted = int(np.sum(optim_res.assignments["AssignedAction"] != "no_intervention"))
    roi_val = (
        (optim_res.total_expected_net_benefit / max(1.0, optim_res.budget_spent)) * 100.0
        if optim_res.budget_spent > 0 else 0.0
    )

    with kpi_col1:
        kpi_card(
            label="BUDGET ALLOCATED",
            value=f"€{optim_res.budget_spent:,.0f}",
            delta=f"{optim_res.budget_spent / max(1.0, optim_res.total_budget):.1%}",
            delta_basis="of cap",
            tooltip=f"Cap: €{optim_res.total_budget:,.0f}",
            delta_direction="positive" if optim_res.budget_spent <= optim_res.total_budget else "negative",
        )
    with kpi_col2:
        kpi_card(
            label="EXPECTED NET BENEFIT",
            value=f"€{optim_res.total_expected_net_benefit:,.0f}",
            delta=f"+{policy_cmp.get('uplift_advantage_pct', 0.0):.1f}%",
            delta_basis="vs baseline",
            tooltip="Discounted retention value net of outreach costs",
            delta_direction="positive",
        )
    with kpi_col3:
        kpi_card(
            label="CUSTOMERS RETAINED",
            value=f"{optim_res.total_expected_incremental_retained:,.1f}",
            delta=f"{total_contacted:,}",
            delta_basis="contacted",
            tooltip="Incremental accounts saved from churn",
            delta_direction="neutral",
        )
    with kpi_col4:
        kpi_card(
            label="PROJECTED CAMPAIGN ROI",
            value=f"{roi_val:.1f}%",
            delta=optim_res.solver_used,
            delta_basis="solver",
            tooltip="Return on retention investment",
            delta_direction="positive" if roi_val > 0 else "negative",
        )

    # Binding constraints callout
    render_html(
        f"""
        <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-left: 4px solid {SECONDARY_ACCENT}; padding: 0.75rem 1rem; border-radius: 8px; margin: 1rem 0;">
            <div style="font-size: 0.75rem; text-transform: uppercase; color: {TEXT_SECONDARY}; font-weight: 600; letter-spacing: 0.05em;">
                OPERATIONAL CONSTRAINT BINDING STATUS
            </div>
            <div style="font-size: 0.88rem; color: {TEXT_PRIMARY}; margin-top: 0.25rem;">
                <b>Binding Constraint(s):</b> {', '.join(optim_res.binding_constraints)}
            </div>
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # POLICY COMPARISON & CAUSAL BENCHMARKS
    # -------------------------------------------------------------------------
    section_header(
        title="Policy Evaluation Benchmark",
        subtitle="Comparing the Prescriptive Causal Optimizer against Standard Industry Churn-Risk Targeting under identical budget.",
    )

    col_bench_chart, col_bench_narrative = st.columns([3, 2])

    with col_bench_chart:
        fig_cmp = build_policy_comparison_bar(policy_cmp["optimizer"], policy_cmp["churn_probability_baseline"])
        st.plotly_chart(fig_cmp, width="stretch")

    with col_bench_narrative:
        render_html(
            f"""
            <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 1.25rem; height: 100%;">
                <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: {TEXT_PRIMARY};">Why Uplift Beats Pure Churn Ranking</h4>
                <p style="font-size: 0.82rem; color: {TEXT_SECONDARY}; line-height: 1.5;">
                    Standard banking retention campaigns typically target the <i>highest churn probability</i> accounts. 
                    However, this strategy is mathematically sub-optimal because it wastes budget on two customer types:
                </p>
                <ul style="font-size: 0.80rem; color: {TEXT_SECONDARY}; line-height: 1.5; padding-left: 1.2rem;">
                    <li><b style="color: {DANGER};">Lost Causes:</b> Extremely disengaged customers who will leave regardless of intervention.</li>
                    <li><b style="color: {WARNING};">Sleeping Dogs:</b> Inactive high-balance accounts where intrusive contact causes annoyance, actually <i>accelerating</i> churn.</li>
                </ul>
                <p style="font-size: 0.82rem; color: {SUCCESS}; line-height: 1.5; margin-top: 0.5rem;">
                    <b>The Prescriptive Optimizer isolates "Persuadables"</b>: customers whose retention probability increases 
                    most per euro of outreach, delivering a <b>+{policy_cmp.get('uplift_advantage_pct', 0.0):.1f}% gain in Net Benefit</b>.
                </p>
            </div>
            """
        )

    # -------------------------------------------------------------------------
    # PAGINATED ASSIGNMENT TABLE & FILTERS
    # -------------------------------------------------------------------------
    section_header(
        title="Customer Intervention Allocations",
        subtitle="Detailed customer-level action assignments optimized for maximum net benefit.",
    )

    assign_df = optim_res.assignments.copy()
    # Merge key attributes for filtering
    assign_df = assign_df.merge(
        df_scored[["CustomerId", "Geography", "RiskCategory", "NumOfProducts", "IsActiveMember"]],
        on="CustomerId",
        how="left",
    )
    assign_df["MaskedId"] = assign_df["CustomerId"].astype(str).map(lambda x: f"ID-***{x[-4:]}")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_acts = st.multiselect(
            "Filter by Assigned Action",
            options=list(assign_df["AssignedAction"].unique()),
            default=[a for a in assign_df["AssignedAction"].unique() if a != "no_intervention"] or ["no_intervention"],
            key="de_filter_act",
        )
    with col_f2:
        min_enb = st.number_input(
            "Minimum Expected Net Benefit (€)",
            value=0.0,
            step=25.0,
            key="de_filter_enb",
        )
    with col_f3:
        selected_risks = st.multiselect(
            "Filter by Risk Category",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="de_filter_risk",
        )

    filtered_assign = assign_df[
        (assign_df["AssignedAction"].isin(selected_acts))
        & (assign_df["ExpectedNetBenefit"] >= min_enb)
        & (assign_df["RiskCategory"].isin(selected_risks))
    ]

    render_html(f"<div style='font-size: 0.82rem; color: {TEXT_SECONDARY}; margin-bottom: 0.5rem;'>Showing <b>{len(filtered_assign):,}</b> customers matching criteria</div>")

    # Format table for executive presentation
    display_cols = [
        "MaskedId", "AssignedAction", "ActionCost", "ExpectedUplift",
        "CLV", "ExpectedNetBenefit", "RiskCategory", "Geography"
    ]
    formatted_df = filtered_assign[display_cols].copy()
    formatted_df["ActionCost"] = formatted_df["ActionCost"].map(lambda x: f"€{x:,.2f}")
    formatted_df["ExpectedUplift"] = formatted_df["ExpectedUplift"].map(lambda x: f"+{x:.1%}")
    formatted_df["CLV"] = formatted_df["CLV"].map(lambda x: f"€{x:,.0f}")
    formatted_df["ExpectedNetBenefit"] = formatted_df["ExpectedNetBenefit"].map(lambda x: f"€{x:,.2f}")

    st.dataframe(formatted_df.head(100), width="stretch", hide_index=True)

    # CSV Download
    csv_bytes = filtered_assign.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Optimized Allocations CSV",
        data=csv_bytes,
        file_name="retention_decision_allocations.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    render_decision_engine()

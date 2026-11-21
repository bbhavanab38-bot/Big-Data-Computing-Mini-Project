"""Retention Campaign ROI & Monte Carlo Uncertainty Simulator Page.

Computes expected campaign economics, break-even retention hurdles, multi-scenario comparisons,
and 5,000-draw Monte Carlo risk simulations with input parameter distributions.
Replaces legacy point-estimate 'optimal N' with a continuous net-benefit curve
featuring a 90% confidence uncertainty ribbon.

Pure Streamlit presentation calling pure-Python modules in dashboard/core/.
"""
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from dashboard.ui.theme import (
    CARD_BG,
    PRIMARY_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from dashboard.ui.components import (
    apply_page_styles,
    empty_state,
    insight_banner,
    kpi_card,
    page_header,
    section_header,
    render_html,
)
from dashboard.ui.charts import (
    build_monte_carlo_distribution_chart,
    build_net_benefit_uncertainty_curve,
    build_tornado_chart,
)
from dashboard.core.roi_simulator import (
    ROIInputs,
    calculate_deterministic_roi,
    compute_net_benefit_curve,
    run_monte_carlo_roi,
)


def render_roi_simulator():
    """Render the ROI and Monte Carlo Simulator interface."""
    apply_page_styles()
    page_header(
        title="Retention ROI & Uncertainty Lab",
        description="Economic projections, scenario modeling, and 5,000-draw Monte Carlo risk analysis.",
        badges=["UNCERTAINTY LAB"],
    )

    # Mandatory Simulation Notice
    insight_banner(
        title="Assumption-Based Planning Model",
        text=(
            "All financial projections, acceptance rates, and uplift parameters are planning assumptions. "
            "Monte Carlo simulations model parametric uncertainty to reveal downside risk and break-even thresholds. "
            "Never treat simulated ROI as guaranteed accounting return."
        ),
        icon="⚠️",
        badge="SIMULATION NOTICE",
    )

    # -------------------------------------------------------------------------
    # SIDEBAR: Campaign Parameters & Economic Assumptions
    # -------------------------------------------------------------------------
    render_html(
        """
        <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 0.25rem;">
            CAMPAIGN ASSUMPTIONS
        </div>
        """,
        container=st.sidebar,
    )

    budget_input = st.sidebar.number_input(
        "Campaign Budget (€)",
        min_value=1000.0,
        max_value=250000.0,
        value=15000.0,
        step=1000.0,
        key="roi_budget",
    )
    offer_cost_input = st.sidebar.number_input(
        "Offer Cost per Contact (€)",
        min_value=5.0,
        max_value=500.0,
        value=50.0,
        step=5.0,
        key="roi_offer_cost",
    )
    capacity_input = st.sidebar.number_input(
        "Operational Outreach Capacity",
        min_value=50,
        max_value=5000,
        value=400,
        step=50,
        key="roi_capacity",
    )
    acceptance_input = st.sidebar.slider(
        "Offer Acceptance Rate (%)",
        min_value=10.0,
        max_value=100.0,
        value=65.0,
        step=1.0,
        key="roi_acceptance",
    ) / 100.0
    clv_input = st.sidebar.number_input(
        "Average Customer CLV (€)",
        min_value=100.0,
        max_value=10000.0,
        value=1450.0,
        step=50.0,
        key="roi_clv",
    )
    uplift_input = st.sidebar.slider(
        "Assumed Retention Uplift (%)",
        min_value=1.0,
        max_value=30.0,
        value=9.0,
        step=0.5,
        key="roi_uplift",
    ) / 100.0

    current_inputs = ROIInputs(
        budget=float(budget_input),
        offer_cost=float(offer_cost_input),
        capacity=int(capacity_input),
        acceptance_rate=float(acceptance_input),
        average_clv=float(clv_input),
        uplift_rate=float(uplift_input),
    )

    # Calculate baseline deterministic outputs
    det_res = calculate_deterministic_roi(current_inputs)

    # -------------------------------------------------------------------------
    # TOP KPI METRICS: Core Financial Return
    # -------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        kpi_card(
            label="TOTAL CAMPAIGN COST",
            value=f"€{det_res.total_cost:,.0f}",
            delta=f"{det_res.contacts_count:,}",
            delta_basis="contacts",
            tooltip=f"Cap: €{current_inputs.budget:,.0f}",
            delta_direction="neutral",
        )
    with col2:
        kpi_card(
            label="EXPECTED RETENTION VALUE",
            value=f"€{det_res.expected_retention_value:,.0f}",
            delta=f"{det_res.incremental_retained_customers:,.1f}",
            delta_basis="retained",
            tooltip="Gross discounted CLV preserved",
            delta_direction="positive",
        )
    with col3:
        kpi_card(
            label="EXPECTED NET BENEFIT",
            value=f"€{det_res.expected_net_benefit:,.0f}",
            delta=f"{det_res.roi_pct:+.1f}%",
            delta_basis="ROI",
            tooltip="Net value creation after costs",
            delta_direction="positive" if det_res.expected_net_benefit > 0 else "negative",
        )
    with col4:
        kpi_card(
            label="CAMPAIGN ROI",
            value=f"{det_res.roi_pct:.1f}%",
            delta="Expected",
            delta_basis="return",
            tooltip="Return on outreach spend",
            delta_direction="positive" if det_res.roi_pct > 0 else "negative",
        )
    with col5:
        kpi_card(
            label="BREAK-EVEN RETENTION RATE",
            value=f"{det_res.break_even_retention_rate:.2%}",
            delta="Hurdle Rate",
            delta_basis="threshold",
            tooltip="Offer Cost / Average CLV",
            delta_direction="neutral",
        )

    # Break-Even Hurdle Formula Display
    render_html(
        f"""
        <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 8px; padding: 0.6rem 1rem; margin: 0.5rem 0 1.25rem 0; font-size: 0.80rem; color: {TEXT_SECONDARY};">
            <b>Break-Even Retention Formula:</b> Minimum required retention rate to cover intervention costs = 
            <code>Offer Cost (€{current_inputs.offer_cost:.0f}) / Average CLV (€{current_inputs.average_clv:.0f}) = {det_res.break_even_retention_rate:.2%}</code>.
            Campaign generates net profit whenever the achieved effective retention exceeds <b>{det_res.break_even_retention_rate:.2%}</b>.
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # TAB NAVIGATION: Continuous Net Benefit vs Monte Carlo vs Scenario Comparison
    # -------------------------------------------------------------------------
    tab_curve, tab_monte_carlo, tab_scenarios = st.tabs([
        "Net Benefit Curve (Continuous)",
        "Monte Carlo Risk Simulation (5,000 Draws)",
        "Scenario Comparison (Up to 3)",
    ])

    # =========================================================================
    # TAB 1: CONTINUOUS NET BENEFIT CURVE (REPLACING "OPTIMAL N")
    # =========================================================================
    with tab_curve:
        section_header(
            title="Continuous Net Benefit & Contact Volume Planning",
            subtitle="Replacing the legacy single 'optimal N' point-estimate with a continuous economics curve and 90% confidence uncertainty ribbon.",
        )

        col_curve_chart, col_curve_info = st.columns([3, 2])

        with col_curve_chart:
            curve_df = compute_net_benefit_curve(
                current_inputs,
                min_contacts=50,
                max_contacts=max(200, current_inputs.capacity),
                num_points=20,
            )
            fig_curve = build_net_benefit_uncertainty_curve(curve_df)
            st.plotly_chart(fig_curve, width="stretch")

        with col_curve_info:
            render_html(
                f"""
                <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 1.25rem; height: 100%;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: {TEXT_PRIMARY};">Why Single "Optimal N" is a Flawed Metric</h4>
                    <p style="font-size: 0.82rem; color: {TEXT_SECONDARY}; line-height: 1.5;">
                        Prior iterations of this project reported a static headline <i>"Optimal N = 1,464"</i>. 
                        In real financial operations, a single point estimate creates a false sense of certainty:
                    </p>
                    <ul style="font-size: 0.80rem; color: {TEXT_SECONDARY}; line-height: 1.5; padding-left: 1.2rem;">
                        <li>CLV and offer uptake fluctuate across customer cohorts and macroeconomic cycles.</li>
                        <li>Diminishing marginal returns and operational outreach constraints create an uncertainty band.</li>
                        <li>Targeting slightly above or below the peak yields near-identical financial performance without risk.</li>
                    </ul>
                    <p style="font-size: 0.82rem; color: {PRIMARY_ACCENT}; line-height: 1.5; margin-top: 0.5rem;">
                        <b>Recommendation:</b> Operate in the plateau region where 90% confidence lower bound (P5) remains 
                        positive, maintaining a buffer against offer rejection and rate variance.
                    </p>
                </div>
                """
            )

    # =========================================================================
    # TAB 2: 5,000-DRAW MONTE CARLO RISK SIMULATION
    # =========================================================================
    with tab_monte_carlo:
        section_header(
            title="Monte Carlo Downside Risk & Sensitivity Analysis",
            subtitle="Evaluating 5,000 independent stochastic trials with parameterized distributions for Acceptance Rate (Beta), CLV (Lognormal), and Uplift (Truncated Normal).",
        )

        col_mc_run, _ = st.columns([1, 4])
        with col_mc_run:
            re_run_mc = st.button("Run 5,000 Stochastic Draws", key="mc_rerun_btn", type="secondary")

        if re_run_mc or "last_mc_result" not in st.session_state:
            with st.spinner("Generating 5,000 Monte Carlo draws..."):
                mc_res = run_monte_carlo_roi(current_inputs, n_draws=5000, seed=42)
                st.session_state["last_mc_result"] = mc_res

        mc_res = st.session_state["last_mc_result"]

        # Monte Carlo Summary KPI Cards
        mc_kpi1, mc_kpi2, mc_kpi3, mc_kpi4 = st.columns(4)
        with mc_kpi1:
            kpi_card(
                label="P(NET BENEFIT > €0)",
                value=f"{mc_res.prob_positive_roi:.1%}",
                delta="Profitable",
                delta_basis="draws",
                tooltip="Probability campaign is net positive",
                delta_direction="positive" if mc_res.prob_positive_roi >= 0.80 else "negative",
            )
        with mc_kpi2:
            kpi_card(
                label="5TH % DOWNSIDE (VaR)",
                value=f"€{mc_res.percentile_5:,.0f}",
                delta="P5 Lower",
                delta_basis="bound",
                tooltip="95% confidence worst-case net outcome",
                delta_direction="neutral" if mc_res.percentile_5 > 0 else "negative",
            )
        with mc_kpi3:
            kpi_card(
                label="MEDIAN BENEFIT (P50)",
                value=f"€{mc_res.percentile_50:,.0f}",
                delta="P50",
                delta_basis="median",
                tooltip="Expected 50th percentile outcome",
                delta_direction="positive",
            )
        with mc_kpi4:
            kpi_card(
                label="95TH % UPSIDE",
                value=f"€{mc_res.percentile_95:,.0f}",
                delta="P95 Bull",
                delta_basis="case",
                tooltip="Top 5% optimistic outcome",
                delta_direction="positive",
            )

        col_hist, col_tornado = st.columns([3, 2])

        with col_hist:
            fig_hist = build_monte_carlo_distribution_chart(mc_res)
            st.plotly_chart(fig_hist, width="stretch")

        with col_tornado:
            fig_tornado = build_tornado_chart(mc_res.tornado_sensitivities)
            st.plotly_chart(fig_tornado, width="stretch")
            st.caption("Spearman correlation indicates which input assumption exerts the greatest influence on net financial outcome.")

    # =========================================================================
    # TAB 3: SCENARIO COMPARISON (UP TO 3 NAMED SCENARIOS)
    # =========================================================================
    with tab_scenarios:
        section_header(
            title="Strategic Scenario Comparison",
            subtitle="Save and benchmark up to three distinct campaign configuration scenarios in session state.",
        )

        if "saved_roi_scenarios" not in st.session_state:
            # Seed with 2 realistic baseline scenarios
            st.session_state["saved_roi_scenarios"] = {
                "Conservative RM Outreach": ROIInputs(
                    budget=12000.0, offer_cost=60.0, capacity=200, acceptance_rate=0.75, average_clv=1800.0, uplift_rate=0.08
                ),
                "Aggressive Digital Campaign": ROIInputs(
                    budget=25000.0, offer_cost=30.0, capacity=800, acceptance_rate=0.55, average_clv=1200.0, uplift_rate=0.06
                ),
            }

        saved_scenarios = st.session_state["saved_roi_scenarios"]

        col_save_name, col_save_btn, col_clear_btn = st.columns([2, 1, 1])
        with col_save_name:
            scenario_name = st.text_input("New Scenario Name", placeholder="e.g. Premium VIP Retainer", key="sc_name_input")
        with col_save_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Save Current Parameters", type="primary", width="stretch"):
                if scenario_name.strip():
                    if len(saved_scenarios) >= 3 and scenario_name.strip() not in saved_scenarios:
                        st.error("Maximum 3 scenarios allowed. Remove an existing scenario first.")
                    else:
                        saved_scenarios[scenario_name.strip()] = current_inputs.model_copy()
                        st.session_state["saved_roi_scenarios"] = saved_scenarios
                        st.success(f"Scenario '{scenario_name.strip()}' saved!")
                        st.rerun()
                else:
                    st.warning("Enter a scenario name.")
        with col_clear_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("Reset Scenarios", type="secondary", width="stretch"):
                st.session_state["saved_roi_scenarios"] = {}
                st.rerun()

        if not saved_scenarios:
            empty_state("No saved scenarios yet. Use the inputs on the left to configure and save a scenario.")
        else:
            # Build comparative dataframe
            cmp_data = []
            for name, sc_inputs in saved_scenarios.items():
                res = calculate_deterministic_roi(sc_inputs)
                mc = run_monte_carlo_roi(sc_inputs, n_draws=1000, seed=42)
                cmp_data.append({
                    "Scenario": name,
                    "Budget": f"€{sc_inputs.budget:,.0f}",
                    "Offer Cost": f"€{sc_inputs.offer_cost:,.0f}",
                    "Contacts": f"{res.contacts_count:,}",
                    "Acceptance": f"{sc_inputs.acceptance_rate:.0%}",
                    "Uplift": f"{sc_inputs.uplift_rate:.1%}",
                    "Expected Cost": f"€{res.total_cost:,.0f}",
                    "Net Benefit": f"€{res.expected_net_benefit:,.0f}",
                    "ROI %": f"{res.roi_pct:+.1f}%",
                    "P(Positive ROI)": f"{mc.prob_positive_roi:.1%}",
                    "Break-Even Rate": f"{res.break_even_retention_rate:.2%}",
                })

            st.dataframe(pd.DataFrame(cmp_data), width="stretch", hide_index=True)

            # Comparative bar chart of Net Benefit
            sc_names = list(saved_scenarios.keys())
            sc_net_benefits = [calculate_deterministic_roi(s).expected_net_benefit for s in saved_scenarios.values()]
            sc_costs = [calculate_deterministic_roi(s).total_cost for s in saved_scenarios.values()]

            fig_sc = go.Figure(data=[
                go.Bar(name="Total Cost (€)", x=sc_names, y=sc_costs, marker_color=TEXT_SECONDARY),
                go.Bar(name="Expected Net Benefit (€)", x=sc_names, y=sc_net_benefits, marker_color=PRIMARY_ACCENT),
            ])
            fig_sc.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_PRIMARY, size=12),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_sc, width="stretch")


if __name__ == "__main__":
    render_roi_simulator()

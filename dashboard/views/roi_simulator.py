"""Retention ROI Simulator view: financial return optimization for targeted campaigns."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.analytics import simulate_retention_roi


def render_roi_simulator(df: pd.DataFrame):
    st.subheader("Retention ROI Simulator")
    st.caption("Optimize targeted retention campaigns based on ranked model predictions and business cost assumptions.")

    st.markdown(
        """> [!NOTE]
        > **Business Assumptions Notice:** Offer cost, customer value, and acceptance rates below are **user-defined business assumptions**, not historical banking data.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        offer_cost = st.number_input(
            "Offer Cost per Customer ($)",
            min_value=5.0,
            max_value=500.0,
            value=50.0,
            step=5.0,
            help="Direct cost of the incentive, voucher, or banker outreach per contacted account.",
        )
    with c2:
        customer_value = st.number_input(
            "Retained Customer Value ($)",
            min_value=50.0,
            max_value=5000.0,
            value=500.0,
            step=50.0,
            help="Estimated annual profit or customer lifetime value protected by preventing churn.",
        )
    with c3:
        success_rate = st.slider(
            "Offer Acceptance Rate (%)",
            min_value=5.0,
            max_value=80.0,
            value=30.0,
            step=1.0,
            help="Assumed percentage of contacted churners who accept the offer and remain with the bank.",
        ) / 100.0

    # Run simulation
    sim_result = simulate_retention_roi(
        df,
        offer_cost=offer_cost,
        customer_value=customer_value,
        success_rate=success_rate,
        step=50,
    )

    curve = sim_result["curve"]
    best_n = sim_result["best_n"]
    max_net = sim_result["max_net_saving"]
    optimal_roi = sim_result["optimal_roi"]

    # KPI Summary Cards
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Optimal Target Size (N*)", f"{best_n:,} accounts", help="Contacting beyond this cutoff yields diminishing returns")
    k2.metric("Max Net Saving", f"${max_net:,.0f}", help="Gross value of saved customers minus outreach expenditure")
    k3.metric("Optimal Campaign ROI", f"{optimal_roi:.1f}%", help="Net savings divided by total outreach budget")
    k4.metric("Campaign Outreach Cost", f"${best_n * offer_cost:,.0f}", help="Total marketing / outreach budget required")

    # Net Savings Chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=curve["N"],
            y=curve["NetSaving"],
            mode="lines",
            name="Expected Net Saving ($)",
            line=dict(color="#2f6655", width=3),
        )
    )
    # Peak point
    fig.add_trace(
        go.Scatter(
            x=[best_n],
            y=[max_net],
            mode="markers+text",
            name="Optimal Cutoff (N*)",
            text=[f"Peak: ${max_net:,.0f} at N={best_n:,}"],
            textposition="top center",
            marker=dict(color="#c76e4e", size=14, symbol="star"),
        )
    )
    # Zero breakeven line
    fig.add_hline(y=0, line_dash="dash", line_color="#888888", annotation_text="Breakeven ($0)")

    fig.update_layout(
        title="Expected Net Financial Saving vs Contact Volume (Ranked by Churn Risk)",
        xaxis_title="Number of Top-Risk Customers Contacted (N)",
        yaxis_title="Net Saving ($)",
        yaxis=dict(tickprefix="$"),
        margin=dict(l=40, r=20, t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("View Simulated Financial Breakdown Table"):
        display_curve = curve.copy()
        display_curve.columns = ["Top N Contacts", "Contact Cost ($)", "Gross Saved ($)", "Net Saving ($)", "ROI (%)"]
        st.dataframe(display_curve, width="stretch", hide_index=True)


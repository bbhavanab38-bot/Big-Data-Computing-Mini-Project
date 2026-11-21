"""Overview (Command Center) Page.

Reference A layout:
  - Top bar with working global search & live system status pill
  - Dynamic insight banner generated from computed statistics
  - 5x2 KPI grid with mathematically honest deltas and basis annotations
  - Risk band distribution next to risk attrition funnel
  - Demographic & behavioral breakdown charts
  - Global reactive sidebar filters
"""
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from dashboard.core.platform_data import (
    load_scored_dataset,
    load_model_comparison_data,
    load_active_model_bundle,
)
from dashboard.core.data import filter_platform_customers
from dashboard.ui.components import (
    apply_page_styles,
    page_header,
    kpi_card,
    insight_banner,
    section_header,
    top_bar_system_status,
    render_html,
)
from dashboard.ui.charts import (
    create_donut_chart,
    create_bar_chart,
    create_funnel_chart,
    create_histogram,
    build_activity_over_time_chart,
)
from dashboard.core.streaming_engine import load_recent_stream_events
from dashboard.ui.theme import PRIMARY_ACCENT, SECONDARY_ACCENT, WARNING


def render_overview():
    apply_page_styles()

    # Load data
    try:
        raw_df = load_scored_dataset()
        model_cmp = load_model_comparison_data()
        bundle = load_active_model_bundle()
        model_loaded = bundle is not None
        data_loaded = not raw_df.empty
    except Exception as e:
        st.error(f"Error loading system artifacts: {e}")
        return

    # Top Bar: Title & Global Search / Status Pill
    col_title, col_status = st.columns([2.8, 1.2])
    with col_title:
        page_header(
            "Portfolio Retention Command Center",
            "Real-time monitoring of calibrated customer churn risk, segments, and capital exposure.",
        )
    with col_status:
        render_html(top_bar_system_status(model_loaded, data_loaded, total_rows=len(raw_df)))

    # -------------------------------------------------------------------------
    # Global Sidebar Filters
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### FILTERS")

    # Geography filter
    all_geos = sorted(raw_df["Geography"].dropna().unique().tolist())
    sel_geos = st.sidebar.multiselect("Geography", all_geos, default=all_geos)

    # Gender filter
    all_genders = sorted(raw_df["Gender"].dropna().unique().tolist())
    sel_genders = st.sidebar.multiselect("Gender", all_genders, default=all_genders)

    # Risk Band filter
    all_risks = ["Low", "Medium", "High"]
    sel_risks = st.sidebar.multiselect("Risk Band", all_risks, default=all_risks)

    # Segment filter
    all_segs = sorted(raw_df["SegmentName"].dropna().unique().tolist())
    sel_segs = st.sidebar.multiselect("Customer Segment", all_segs, default=all_segs)

    # Age Range filter
    min_age, max_age = int(raw_df["Age"].min()), int(raw_df["Age"].max())
    sel_age = st.sidebar.slider("Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))

    # Apply filters dynamically
    df = filter_platform_customers(
        raw_df,
        geographies=sel_geos,
        genders=sel_genders,
        risk_categories=sel_risks,
        segments=sel_segs,
        age_range=sel_age,
    )

    if df.empty:
        st.warning("No customers match the current filter criteria. Please broaden your selection.")
        return

    # -------------------------------------------------------------------------
    # Dynamic Insight Banner (Template-derived from computed stats)
    # -------------------------------------------------------------------------
    total_cust = len(df)
    obs_churn_rate = (df["Exited"].mean() * 100.0) if "Exited" in df.columns else 0.0

    # Find highest churn geography
    geo_stats = df.groupby("Geography")["Exited"].mean() * 100.0 if "Exited" in df.columns else pd.Series()
    top_geo = geo_stats.idxmax() if not geo_stats.empty else "N/A"
    top_geo_rate = geo_stats.max() if not geo_stats.empty else 0.0
    geo_diff = top_geo_rate - obs_churn_rate

    # Find segment with highest high-risk concentration
    seg_high_risk = df[df["RiskCategory"] == "High"].groupby("SegmentName")["CustomerId"].count()
    top_risk_seg = seg_high_risk.idxmax() if not seg_high_risk.empty else "Inactive High-Balance Disengaged"
    top_risk_count = seg_high_risk.max() if not seg_high_risk.empty else 0

    insight_banner(
        "Key Portfolio Finding",
        f"{top_geo} exhibits the highest regional churn rate at {top_geo_rate:.1f}% (+{geo_diff:.1f}% vs filtered avg). "
        f"Segment '{top_risk_seg}' concentrates {top_risk_count:,} high-risk customers, driving primary deposit hazard.",
        icon="⚡",
        badge="AUTOMATED AUDIT",
    )

    # -------------------------------------------------------------------------
    # 5x2 KPI Grid
    # -------------------------------------------------------------------------
    pred_churners = int((df["PredictedChurn"] == 1).sum()) if "PredictedChurn" in df.columns else 0
    high_risk_cust = int((df["RiskCategory"] == "High").sum())
    avg_cal_prob = float(df["CalibratedProbability"].mean())
    total_clv = float(df["EstimatedCLV"].sum())

    # Row 1
    r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns(5)
    with r1c1:
        kpi_card(
            "Total Customers",
            f"{total_cust:,}",
            delta=f"{total_cust/len(raw_df):.0%}",
            delta_basis="of portfolio",
            tooltip="Active records matching current filters.",
            delta_direction="neutral",
        )
    with r1c2:
        baseline_rate = raw_df["Exited"].mean() * 100.0
        diff_churn = obs_churn_rate - baseline_rate
        kpi_card(
            "Observed Churn Rate",
            f"{obs_churn_rate:.1f}%",
            delta=f"{diff_churn:+.1f}%",
            delta_basis="vs 16.4% baseline",
            tooltip="Empirical churn rate measured from historical records.",
            delta_direction="negative_is_good",
        )
    with r1c3:
        kpi_card(
            "Predicted Churners",
            f"{pred_churners:,}",
            delta=f"{pred_churners/total_cust:.1%}",
            delta_basis="threshold 0.50",
            tooltip="Customers with calibrated probability >= 50%.",
            delta_direction="negative_is_good",
        )
    with r1c4:
        kpi_card(
            "High-Risk Customers",
            f"{high_risk_cust:,}",
            delta=f"{high_risk_cust/total_cust:.1%}",
            delta_basis="hazard >= 25%",
            tooltip="Customers in elevated danger zone (calibrated probability >= 25%).",
            delta_direction="negative_is_good",
        )
    with r1c5:
        kpi_card(
            "Avg Calibrated Prob",
            f"{avg_cal_prob:.1%}",
            delta=f"{avg_cal_prob - 0.164:+.1%}",
            delta_basis="vs baseline 16.4%",
            tooltip="Mean calibrated probability from 5-fold cross-validated / holdout inference.",
            delta_direction="negative_is_good",
        )

    # Row 2
    r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
    with r2c1:
        kpi_card(
            "Estimated CLV",
            f"${total_clv/1e6:.1f}M",
            tooltip="Estimated Customer Lifetime Value: Annual Margin / (Calibrated Prob + 8% Discount).",
            delta_direction="neutral",
        )
    with r2c2:
        kpi_card(
            "Model ROC-AUC",
            "0.7295",
            delta="+0.2295",
            delta_basis="vs chance 0.50",
            tooltip="Test split discrimination capability (theoretical ceiling ~0.748).",
            delta_direction="positive",
        )
    with r2c3:
        kpi_card(
            "Model Brier Score",
            "0.1216",
            delta="-40.1%",
            delta_basis="vs uncalibrated 0.203",
            tooltip="Mean squared error on probabilities (perfect = 0.0).",
            delta_direction="positive",
        )
    with r2c4:
        high_risk_bal = df[df["RiskCategory"] == "High"]["Balance"].sum()
        kpi_card(
            "High-Risk Balance",
            f"${high_risk_bal/1e6:.1f}M",
            delta=f"{high_risk_bal / df['Balance'].sum():.1%}",
            delta_basis="of filtered deposits",
            tooltip="Total deposit balance held by customers flagged as High Risk.",
            delta_direction="negative_is_good",
        )
    with r2c5:
        kpi_card(
            "Active Segments",
            f"{df['Segment'].nunique()}",
            delta="4 Personas",
            delta_basis="K-Means (k=4)",
            tooltip="Behavioral personas dynamically derived from cluster z-scores.",
            delta_direction="neutral",
        )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Main Visual: Risk Bands & Portfolio Funnel (Reference A layout)
    # -------------------------------------------------------------------------
    col_chart, col_funnel = st.columns([1.8, 1.2])

    with col_chart:
        section_header("Risk Band Distribution & Churn Reality", "Observed churn rate and volume across calibrated risk tiers.")
        # Aggregate risk band stats
        risk_summary = df.groupby("RiskCategory", observed=False).agg(
            Count=("CustomerId", "count"),
            ObservedRate=("Exited", lambda x: (x.mean() * 100.0) if len(x) > 0 else 0.0),
        ).reindex(["Low", "Medium", "High"]).fillna(0)

        fig_risk = create_bar_chart(
            x=["Low (<10%)", "Medium (10-25%)", "High (>=25%)"],
            y=risk_summary["ObservedRate"].round(1).tolist(),
            title="Observed Churn Rate by Calibrated Risk Band (%)",
            color=PRIMARY_ACCENT,
            y_title="Observed Churn Rate (%)",
            text_template="%{y:.1f}%",
        )
        st.plotly_chart(fig_risk, width="stretch")

    with col_funnel:
        section_header("Portfolio Risk Attrition Funnel", "Progression from total portfolio to high-probability churners.")
        med_risk_count = int((df["RiskCategory"].isin(["Medium", "High"])).sum())
        stages = [
            "Total Portfolio",
            "Scored Customers",
            "Elevated/High Risk (>=10%)",
            "Critical High Risk (>=25%)",
            "Predicted Churners (>=50%)",
        ]
        funnel_vals = [
            len(raw_df),
            total_cust,
            med_risk_count,
            high_risk_cust,
            pred_churners,
        ]
        fig_funnel = create_funnel_chart(stages, funnel_vals)
        st.plotly_chart(fig_funnel, width="stretch")

    # -------------------------------------------------------------------------
    # Further Demographic & Behavioral Charts
    # -------------------------------------------------------------------------
    section_header("Cohort Breakdown & Risk Drivers", "Demographic, product, and behavioral distributions across filtered accounts.")

    c1, c2, c3 = st.columns(3)
    with c1:
        # Churn by Geography
        geo_agg = df.groupby("Geography", as_index=False)["Exited"].mean()
        geo_agg["Rate"] = (geo_agg["Exited"] * 100.0).round(1)
        fig_geo = create_bar_chart(
            x=geo_agg["Geography"].tolist(),
            y=geo_agg["Rate"].tolist(),
            title="Churn Rate by Country (%)",
            color=PRIMARY_ACCENT,
            y_title="Churn Rate (%)",
            text_template="%{y:.1f}%",
        )
        st.plotly_chart(fig_geo, width="stretch")

    with c2:
        # Churn by Gender
        gender_agg = df.groupby("Gender", as_index=False)["Exited"].mean()
        gender_agg["Rate"] = (gender_agg["Exited"] * 100.0).round(1)
        fig_gender = create_bar_chart(
            x=gender_agg["Gender"].tolist(),
            y=gender_agg["Rate"].tolist(),
            title="Churn Rate by Gender (%)",
            color=SECONDARY_ACCENT,
            y_title="Churn Rate (%)",
            text_template="%{y:.1f}%",
        )
        st.plotly_chart(fig_gender, width="stretch")

    with c3:
        # Segment Distribution
        seg_counts = df["SegmentName"].value_counts()
        fig_seg = create_donut_chart(
            labels=seg_counts.index.tolist(),
            values=seg_counts.values.tolist(),
            title="Segment Share",
        )
        st.plotly_chart(fig_seg, width="stretch")

    c4, c5 = st.columns(2)
    with c4:
        # Churn by Age Group
        df["AgeBand"] = pd.cut(
            df["Age"],
            [17, 25, 35, 45, 55, 65, 120],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        age_agg = df.groupby("AgeBand", observed=False, as_index=False)["Exited"].mean()
        age_agg["Rate"] = (age_agg["Exited"] * 100.0).round(1)
        fig_age = create_bar_chart(
            x=age_agg["AgeBand"].astype(str).tolist(),
            y=age_agg["Rate"].tolist(),
            title="Churn Rate by Age Group (%)",
            color=WARNING,
            y_title="Churn Rate (%)",
            text_template="%{y:.1f}%",
        )
        st.plotly_chart(fig_age, width="stretch")

    with c5:
        # Customer Activity Score Distribution
        fig_act = create_histogram(
            df["CustomerActivityScore"].dropna().tolist(),
            title="Customer Activity Score Distribution",
            nbins=20,
            color=PRIMARY_ACCENT,
            x_title="CustomerActivityScore (0.0 to 1.0)",
        )
        st.plotly_chart(fig_act, width="stretch")

    # -------------------------------------------------------------------------
    # Simulated Event Activity Over Time
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        "Simulated Interaction Activity Over Time",
        "Longitudinal transaction, login, and balance event volume synthesized by the streaming generator (strictly simulated; no historical timestamps exist in the static dataset).",
    )
    events_df = load_recent_stream_events(limit=500)
    fig_activity = build_activity_over_time_chart(events_df, title="Daily Customer Activity Events (Simulated Stream)")
    st.plotly_chart(fig_activity, width="stretch")


if __name__ == "__main__":
    render_overview()


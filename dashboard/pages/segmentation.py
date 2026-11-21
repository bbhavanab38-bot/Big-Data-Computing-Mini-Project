"""Customer Segmentation Page.

Reference C style:
  - Cluster share distribution & high-level persona metrics
  - Segment profile table with in-cell bar visualizers
  - Radar chart and multi-dimensional behavioral comparison
  - Rule-derived persona descriptions and recommended retention strategies
  - Distributed Spark MLlib K-Means benchmark comparison
"""
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.core.platform_data import (
    load_scored_dataset,
    load_spark_platform_data,
    load_active_model_bundle,
)
from dashboard.core.segments import (
    build_segment_profiles,
    get_normalized_radar_metrics,
)
from dashboard.ui.components import (
    apply_page_styles,
    page_header,
    section_header,
    top_bar_system_status,
    insight_banner,
    render_html,
)
from dashboard.ui.charts import (
    create_donut_chart,
    create_radar_chart,
)
from dashboard.ui.theme import (
    CARD_BG,
    BORDER_COLOR,
    PRIMARY_ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)


def render_segmentation():
    apply_page_styles()

    raw_df = load_scored_dataset()
    spark_summary, _ = load_spark_platform_data()
    bundle = load_active_model_bundle()
    model_loaded = bundle is not None
    data_loaded = not raw_df.empty

    col_title, col_status = st.columns([2.8, 1.2])
    with col_title:
        page_header(
            "Behavioral Customer Segmentation",
            "Multi-dimensional cluster profiling, rule-derived retention personas, and distributed Spark MLlib alignment.",
        )
    with col_status:
        render_html(top_bar_system_status(model_loaded, data_loaded, total_rows=len(raw_df)))

    # Compute current empirical segment summary
    seg_agg = raw_df.groupby("Segment", as_index=False).agg(
        Customers=("CustomerId", "count"),
        AvgAge=("Age", "mean"),
        AvgBalance=("Balance", "mean"),
        AvgCreditScore=("CreditScore", "mean"),
        AvgSalary=("EstimatedSalary", "mean"),
        AvgActivity=("CustomerActivityScore", "mean"),
        ChurnRatePct=("Exited", lambda x: float(x.mean() * 100.0)),
    )
    seg_agg["AvgAge"] = seg_agg["AvgAge"].round(1)
    seg_agg["AvgBalance"] = seg_agg["AvgBalance"].round(0)
    seg_agg["AvgCreditScore"] = seg_agg["AvgCreditScore"].round(1)
    seg_agg["AvgSalary"] = seg_agg["AvgSalary"].round(0)
    seg_agg["AvgActivity"] = seg_agg["AvgActivity"].round(3)
    seg_agg["ChurnRatePct"] = seg_agg["ChurnRatePct"].round(2)

    profiles = build_segment_profiles(seg_agg)

    # 4 Persona KPI Cards
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    persona_cols = [col_p1, col_p2, col_p3, col_p4]

    for idx, (_, row) in enumerate(profiles.iterrows()):
        with persona_cols[idx % 4]:
            seg_id = int(row["Segment"])
            risk_color = DANGER if row["RiskLevel"] == "Critical" else (WARNING if row["RiskLevel"] in ["Elevated", "Moderate"] else SUCCESS)
            render_html(
                f"""
                <div class="retention-card" style="border-top: 3px solid {risk_color}; height: 100%;">
                    <div style="font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: {TEXT_SECONDARY};">Segment {seg_id} • {row['RiskLevel']} Risk</div>
                    <div style="font-size: 1rem; font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.25rem; line-height: 1.2;">
                        {row['SegmentName']}
                    </div>
                    <div style="margin-top: 0.65rem; display: flex; justify-content: space-between; font-size: 0.8rem;">
                        <span style="color: {TEXT_SECONDARY};">Customers:</span>
                        <span style="font-weight: 600; color: {TEXT_PRIMARY};">{row['Customers']:,} ({row['CustomerSharePct']:.1f}%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.2rem;">
                        <span style="color: {TEXT_SECONDARY};">Churn Rate:</span>
                        <span style="font-weight: 700; color: {risk_color};">{row['ChurnRatePct']:.1f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 0.2rem;">
                        <span style="color: {TEXT_SECONDARY};">Avg Balance:</span>
                        <span style="font-weight: 600; color: {TEXT_PRIMARY};">${row['AvgBalance']:,.0f}</span>
                    </div>
                </div>
                """
            )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Visual Comparisons: Segment Share Donut & Radar Chart (Reference C style)
    # -------------------------------------------------------------------------
    col_donut, col_radar = st.columns([1.2, 1.8])

    with col_donut:
        section_header("Cluster Portfolio Share", "Distribution of customers across the 4 clusters.")
        fig_donut = create_donut_chart(
            labels=profiles["SegmentName"].tolist(),
            values=profiles["Customers"].tolist(),
            hole=0.6,
        )
        st.plotly_chart(fig_donut, width="stretch")

    with col_radar:
        section_header("Multi-Dimensional Persona Radar", "Normalized 0-1 metrics across key banking dimensions.")
        radar_df = get_normalized_radar_metrics(profiles)
        if not radar_df.empty:
            metrics_list = ["Age", "Balance", "Activity", "Churn Rate (%)"]
            series_dict = {}
            for seg_label in radar_df["Segment"].unique():
                sub = radar_df[radar_df["Segment"] == seg_label]
                series_dict[seg_label.split(":")[0]] = sub["NormalizedScore"].tolist()

            fig_radar = create_radar_chart(metrics_list, series_dict)
            st.plotly_chart(fig_radar, width="stretch")

    # -------------------------------------------------------------------------
    # Detailed Persona Profile Table with In-Cell Bar Comparisons
    # -------------------------------------------------------------------------
    section_header("Segment Behavioral Benchmark Table", "Empirical comparison with in-cell distribution metrics.")

    # Render styled custom table
    table_rows = []
    max_bal = profiles["AvgBalance"].max()
    max_churn = profiles["ChurnRatePct"].max()

    for _, r in profiles.iterrows():
        bal_pct = (r["AvgBalance"] / max_bal * 100.0) if max_bal > 0 else 0
        churn_pct = (r["ChurnRatePct"] / max_churn * 100.0) if max_churn > 0 else 0
        churn_color = DANGER if r["ChurnRatePct"] >= 20.0 else (WARNING if r["ChurnRatePct"] >= 15.0 else SUCCESS)

        table_rows.append(
            f"""
            <tr style="border-bottom: 1px solid rgba(148, 163, 184, 0.12);">
                <td style="padding: 0.75rem 0.5rem; font-weight: 600; color: {TEXT_PRIMARY};">
                    <div>Seg {r['Segment']}: {r['SegmentName']}</div>
                    <div style="font-size: 0.72rem; color: {TEXT_SECONDARY};">{r['Description'][:75]}...</div>
                </td>
                <td style="padding: 0.75rem 0.5rem; color: {TEXT_PRIMARY}; font-weight: 600;">{r['Customers']:,}</td>
                <td style="padding: 0.75rem 0.5rem; color: {TEXT_PRIMARY};">{r['CustomerSharePct']:.1f}%</td>
                <td style="padding: 0.75rem 0.5rem; color: {TEXT_PRIMARY};">{r['AvgAge']:.1f} yrs</td>
                <td style="padding: 0.75rem 0.5rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;">
                        <span style="color: {TEXT_PRIMARY};">${r['AvgBalance']:,.0f}</span>
                        <div class="bar-container" style="max-width: 65px;"><div class="bar-fill" style="width: {bal_pct:.1f}%; background-color: {PRIMARY_ACCENT};"></div></div>
                    </div>
                </td>
                <td style="padding: 0.75rem 0.5rem; color: {TEXT_PRIMARY};">{r['AvgActivity']:.3f}</td>
                <td style="padding: 0.75rem 0.5rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; gap: 0.5rem;">
                        <span style="color: {churn_color}; font-weight: 700;">{r['ChurnRatePct']:.1f}%</span>
                        <div class="bar-container" style="max-width: 65px;"><div class="bar-fill" style="width: {churn_pct:.1f}%; background-color: {churn_color};"></div></div>
                    </div>
                </td>
            </tr>
            """
        )

    render_html(
        f"""
        <div style="overflow-x: auto; background: {CARD_BG}; border: 1px solid {BORDER_COLOR}; border-radius: 12px; padding: 0.5rem 1rem;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.82rem; text-align: left;">
                <thead>
                    <tr style="border-bottom: 2px solid rgba(148, 163, 184, 0.2); color: {TEXT_SECONDARY}; text-transform: uppercase; font-size: 0.70rem; letter-spacing: 0.05em;">
                        <th style="padding: 0.6rem 0.5rem;">Segment Persona</th>
                        <th style="padding: 0.6rem 0.5rem;">Customers</th>
                        <th style="padding: 0.6rem 0.5rem;">Share</th>
                        <th style="padding: 0.6rem 0.5rem;">Avg Age</th>
                        <th style="padding: 0.6rem 0.5rem;">Avg Balance</th>
                        <th style="padding: 0.6rem 0.5rem;">Activity Score</th>
                        <th style="padding: 0.6rem 0.5rem;">Churn Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
        """
    )

    # -------------------------------------------------------------------------
    # Prescriptive Strategy Cards
    # -------------------------------------------------------------------------
    section_header("Rule-Derived Retention Strategies", "Algorithmic recommendations tailored to each cluster's behavioral profile.")

    c_s1, c_s2 = st.columns(2)
    for idx, (_, r) in enumerate(profiles.iterrows()):
        target_col = c_s1 if idx % 2 == 0 else c_s2
        with target_col:
            render_html(
                f"""
                <div class="retention-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; color: {PRIMARY_ACCENT}; font-size: 0.95rem;">Seg {r['Segment']}: {r['SegmentName']}</span>
                        <span class="risk-chip {'risk-high' if r['RiskLevel'] == 'Critical' else ('risk-medium' if r['RiskLevel'] in ['Elevated', 'Moderate'] else 'risk-low')}">{r['RiskLevel']}</span>
                    </div>
                    <div style="font-size: 0.80rem; color: {TEXT_SECONDARY}; margin-top: 0.4rem; line-height: 1.4;">
                        <strong>Persona Profile:</strong> {r['Description']}
                    </div>
                    <div style="background: rgba(56, 189, 248, 0.07); border-radius: 6px; padding: 0.6rem 0.8rem; margin-top: 0.6rem; font-size: 0.80rem; color: {TEXT_PRIMARY}; border-left: 3px solid {PRIMARY_ACCENT};">
                        <strong>Recommended Strategy:</strong> {r['RecommendedStrategy']}
                    </div>
                </div>
                """
            )

    # -------------------------------------------------------------------------
    # Spark MLlib Alignment Section
    # -------------------------------------------------------------------------
    section_header("Distributed Spark MLlib K-Means Alignment", "Verification of customer-level scikit-learn segmentation against Spark on YARN clustering.")

    insight_banner(
        title="Distributed Spark MLlib Pipeline",
        text=(
            "Customer-level clustering mirrors the distributed Spark MLlib pipeline (spark/spark_preprocessing.py). "
            "Both execute standard scaling on the exact same 6 features (Age, Balance, CreditScore, TransactionCount, EstimatedSalary, CustomerActivityScore) "
            "with k=4 and seed=42, preserving mathematical consistency between Big Data batch processing and interactive inference."
        ),
        badge="SPARK INTEGRATION",
        icon="⚡",
    )

    if spark_summary is not None and not spark_summary.empty:
        with st.expander("View Partitioned Spark Cluster Export Output", expanded=False):
            st.dataframe(spark_summary, width="stretch")


if __name__ == "__main__":
    render_segmentation()

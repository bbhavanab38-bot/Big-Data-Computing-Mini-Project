"""Segment profiles view: Spark K-Means cluster archetypes, radar comparisons, and strategies."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.segments import build_segment_profiles, get_normalized_radar_metrics


def render_segment_profiles(segment_summary_df: pd.DataFrame):
    st.subheader("Spark K-Means Behavioral Segments")
    st.caption("Distributed clustering on YARN: archetypes mathematically derived from measured empirical averages.")

    if segment_summary_df is None or segment_summary_df.empty:
        st.error("Spark K-Means segment summary data unavailable.")
        return

    profiles_df = build_segment_profiles(segment_summary_df)

    # Overview Table
    st.markdown("#### Segment Profile Matrix")
    display_tbl = profiles_df[[
        "Segment", "SegmentName", "Customers", "CustomerSharePct",
        "AvgAge", "AvgBalance", "AvgActivity", "ChurnRatePct", "RiskLevel"
    ]].copy()
    display_tbl.columns = [
        "Segment ID", "Archetype Name", "Customers", "Share (%)",
        "Avg Age (Yrs)", "Avg Balance ($)", "Activity Score", "Churn Rate (%)", "Risk Category"
    ]
    st.dataframe(display_tbl, width="stretch", hide_index=True)

    st.divider()

    # Visualizations: Radar Chart & Churn Comparison
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Cluster Dimensionality (Radar Comparison)")
        radar_df = get_normalized_radar_metrics(profiles_df)

        fig_radar = go.Figure()
        categories = ["Avg Age", "Avg Balance", "Avg Activity", "Churn Rate (%)"]

        colors = ["#c76e4e", "#2f6655", "#5b8975", "#d2a35f"]
        for idx, (seg_name, group) in enumerate(radar_df.groupby("Segment")):
            # Align values with categories
            val_map = dict(zip(group["Metric"], group["NormalizedScore"]))
            vals = [val_map.get(cat, 0.5) for cat in categories]
            vals.append(vals[0])  # Close radar loop

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=vals,
                    theta=categories + [categories[0]],
                    name=seg_name,
                    line=dict(color=colors[idx % len(colors)], width=2),
                    fill="toself",
                    opacity=0.2,
                )
            )

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10)),
            ),
            margin=dict(l=40, r=40, t=30, b=30),
            height=380,
            showlegend=True,
        )
        st.plotly_chart(fig_radar, width="stretch")

    with right:
        st.markdown("#### Churn Rate by Segment Archetype")
        fig_bar = px.bar(
            profiles_df,
            x="SegmentName",
            y="ChurnRatePct",
            color="RiskLevel",
            color_discrete_map={
                "Critical": "#d9534f",
                "Elevated": "#d9a441",
                "Moderate": "#e2b865",
                "Low": "#3a9d7d",
            },
            title="Measured Churn Hazard (%)",
            labels={"SegmentName": "Cluster Archetype", "ChurnRatePct": "Churn Rate (%)"},
            text="ChurnRatePct",
        )
        fig_bar.update_traces(texttemplate="%{text:.2f}%", hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
        fig_bar.update_yaxes(ticksuffix="%", tickformat=".1f")
        fig_bar.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=80))
        st.plotly_chart(fig_bar, width="stretch")

    st.divider()

    # Strategic Action Cards
    st.markdown("#### Banking Retention Playbooks by Archetype")
    c1, c2 = st.columns(2)

    for idx, (_, row) in enumerate(profiles_df.iterrows()):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            st.markdown(
                f"""<div style="background:#ffffff; border:1px solid #dfe5df; border-top:3px solid {'#d9534f' if row['RiskLevel']=='Critical' else '#2f6655'}; border-radius:8px; padding:16px; margin-bottom:16px;">
                    <div style="font-size:12px; font-weight:700; color:#5c7068; text-transform:uppercase;">Segment {row['Segment']} • {row['RiskLevel']} Risk</div>
                    <h4 style="margin:4px 0 8px 0; color:#182e27;">{row['SegmentName']}</h4>
                    <p style="font-size:13px; color:#384c45; margin-bottom:8px;">{row['Description']}</p>
                    <div style="background:#f4f7f4; padding:8px 12px; border-radius:4px; font-size:12px; color:#203c33;">
                        <strong>Recommended Strategy:</strong> {row['RecommendedStrategy']}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )


"""Executive summary export view: downloadable standalone HTML briefing."""
import pandas as pd
import streamlit as st

from dashboard.core.report_generator import generate_executive_report_html


def render_executive_summary(
    kpis: dict,
    python_models: pd.DataFrame,
    spark_models: pd.DataFrame,
    segment_profiles: pd.DataFrame,
    top_drivers: pd.DataFrame,
    high_risk_count: int,
):
    st.subheader("Executive Briefing & Report Export")
    st.caption("Generate and export a comprehensive, print-ready HTML one-page executive summary.")

    html_report = generate_executive_report_html(
        kpis=kpis,
        python_models=python_models,
        spark_models=spark_models,
        segment_profiles=segment_profiles,
        top_drivers=top_drivers,
        high_risk_count=high_risk_count,
    )

    st.markdown(
        """<div style="background:#f6f9f6; border:1px solid #dbe3db; border-radius:8px; padding:20px; margin-bottom:20px;">
            <h4 style="margin:0 0 8px 0; color:#182e27;">One-Page Executive Intelligence Document</h4>
            <p style="font-size:13px; color:#3b524a; margin:0 0 16px 0;">
                Self-contained HTML report ready for distribution or offline presentation. Includes high-level KPIs, full model comparison benchmarks (scikit-learn & Spark on YARN separated), top permutation drivers, and Spark K-Means behavioral playbooks.
            </p>
        </div>""",
        unsafe_allow_html=True,
    )

    btn_col, _ = st.columns([2, 2])
    with btn_col:
        st.download_button(
            label="📥 Download Executive Summary (HTML)",
            data=html_report,
            file_name="banking_churn_executive_summary.html",
            mime="text/html",
            help="Download printable HTML report built directly from current verified data.",
        )

    st.divider()
    st.markdown("#### Live Report Preview")
    st.html(html_report)

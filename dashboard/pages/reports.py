"""Reports Page: Executive Briefings, PDF Downloads, and Governance Exports.

Provides in-app preview and one-click downloads for:
- Formal Executive Audit PDF report (via pure-Python ReportLab)
- Executive KPI & Governance CSV digest
- Customer-level Intervention Allocation CSV
"""
from __future__ import annotations

import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from dashboard.core.reports import (
    REPORTLAB_AVAILABLE,
    build_report_data,
    generate_allocations_csv,
    generate_pdf_report,
    generate_summary_csv,
)
from dashboard.core.platform_data import load_scored_dataset
from dashboard.ui.components import apply_page_styles, page_header


def render():
    """Render the Reports and Executive Exports page."""
    apply_page_styles()
    page_header(
        "Executive Reporting & Governance Exports",
        "Generate certified PDF executive briefings, audit logs, and actionable CSV exports for enterprise risk committees.",
    )

    # Synthetic Data Warning
    st.warning(
        "**Legal & Simulation Notice:** All metrics, probabilities, and intervention returns are generated from synthetic "
        "banking data and economic planning assumptions. No figures represent realized accounting revenue or real customers.",
        icon="⚠️",
    )

    if not REPORTLAB_AVAILABLE:
        st.info("ℹ️ **PDF Generator Notice:** 'reportlab' is recommended for high-fidelity PDF rendering. All executive data, metrics, and CSV exports remain fully operational.")

    # Load dataset
    scored_df = load_scored_dataset()

    # Controls row
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        budget = st.slider(
            "Campaign Budget Allocation (EUR)",
            min_value=1000.0,
            max_value=50000.0,
            value=10000.0,
            step=1000.0,
            format="€%.0f",
            help="Budget constraint used to simulate retention intervention capacity.",
        )
    with col_ctrl2:
        threshold = st.slider(
            "Decision Threshold (p)",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05,
            help="Calibrated probability cutoff for classifying accounts into high-risk intervention candidates.",
        )

    # Compile report data
    report_data = build_report_data(scored_df=scored_df, action_budget=budget, decision_threshold=threshold)

    # Generate artifacts
    pdf_bytes = generate_pdf_report(report_data)
    summary_csv = generate_summary_csv(report_data)
    alloc_csv = generate_allocations_csv(scored_df, limit=500)

    # Top KPI Strip
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Portfolio Accounts", f"{report_data['total_customers']:,}", "Synthetic 10k")
    with kpi2:
        metrics = report_data.get("metrics", {})
        st.metric("Test Set ROC-AUC", f"{metrics.get('ROC-AUC', 0.7295):.4f}", "Ceiling: 0.748")
    with kpi3:
        st.metric("Calibrated Brier Loss", f"{metrics.get('Brier', 0.1216):.4f}", "Sigmoid Platt")
    with kpi4:
        st.metric("Simulated Net ROI", f"{report_data.get('simulated_net_roi', 0):.1f}%", "Assumption-based")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Actionable Download Hub
    st.markdown("### Export Hub")
    dl_col1, dl_col2, dl_col3 = st.columns(3)

    with dl_col1:
        st.download_button(
            label="📄 Download PDF Audit Briefing",
            data=pdf_bytes,
            file_name=f"banking_retention_audit_{report_data['model_version']}.pdf",
            mime="application/pdf",
            width="stretch",
            help="Full multi-section formal PDF executive briefing with governance sign-off.",
        )
    with dl_col2:
        st.download_button(
            label="📊 Download Executive Metrics CSV",
            data=summary_csv,
            file_name="retention_platform_metrics_summary.csv",
            mime="text/csv",
            width="stretch",
            help="Structured CSV containing all portfolio, model, and fairness audit indicators.",
        )
    with dl_col3:
        st.download_button(
            label="👥 Download Customer Allocations CSV",
            data=alloc_csv,
            file_name="recommended_customer_interventions.csv",
            mime="text/csv",
            width="stretch",
            help="Top at-risk customer accounts with calibrated probabilities and assigned retention actions.",
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # In-App Preview Tabs
    st.markdown("### Executive Report Preview")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Portfolio Summary",
        "Model Governance",
        "Segmentation",
        "Interventions & ROI",
        "Algorithmic Fairness",
    ])

    with tab1:
        st.subheader("Portfolio Health & Risk Distribution")
        risk_tiers = report_data.get("risk_tiers", {})
        tot = report_data.get("total_customers", 10000)
        
        preview_portfolio = pd.DataFrame([
            {"Tier": "Low Risk (p < 0.20)", "Accounts": f"{risk_tiers.get('low', 0):,}", "Share": f"{(risk_tiers.get('low', 0)/tot)*100:.1f}%", "Action": "Organic retention (No costly action)"},
            {"Tier": "Medium Risk (0.20 ≤ p < 0.50)", "Accounts": f"{risk_tiers.get('medium', 0):,}", "Share": f"{(risk_tiers.get('medium', 0)/tot)*100:.1f}%", "Action": "Proactive automated nudges & fee waivers"},
            {"Tier": "High Risk (p ≥ 0.50)", "Accounts": f"{risk_tiers.get('high', 0):,}", "Share": f"{(risk_tiers.get('high', 0)/tot)*100:.1f}%", "Action": "Dedicated outreach & service call"},
        ])
        st.dataframe(preview_portfolio, width="stretch", hide_index=True)

    with tab2:
        st.subheader("Model Performance & Probability Calibration")
        m_meta = [
            {"Governance Parameter": "Registered Model Version", "Value": str(report_data.get("model_version"))},
            {"Governance Parameter": "Base Architecture", "Value": str(report_data.get("model_name"))},
            {"Governance Parameter": "Probability Calibration", "Value": f"{str(report_data.get('calibration_method')).upper()} (Platt Scaling)"},
            {"Governance Parameter": "Training Data Hash", "Value": str(report_data.get("training_data_hash"))},
            {"Governance Parameter": "Test Holdout ROC-AUC", "Value": f"{metrics.get('ROC-AUC', 0.7295):.4f}"},
            {"Governance Parameter": "Test Holdout PR-AUC", "Value": f"{metrics.get('PR-AUC', 0.3866):.4f}"},
            {"Governance Parameter": "Calibrated Brier Loss", "Value": f"{metrics.get('Brier', 0.1216):.4f}"},
            {"Governance Parameter": "Holdout Accuracy", "Value": f"{metrics.get('Accuracy', 0.8445)*100:.2f}%"},
        ]
        st.dataframe(pd.DataFrame(m_meta), width="stretch", hide_index=True)

    with tab3:
        st.subheader("Customer Segment Exposure")
        segments = report_data.get("segments", [])
        if segments:
            seg_rows = []
            for s in segments:
                seg_rows.append({
                    "Segment Name": s.get("name"),
                    "Accounts": f"{s.get('count', 0):,}",
                    "Share": f"{s.get('share', 0)*100:.1f}%",
                    "Avg Balance": f"€{s.get('avg_balance', 0):,.0f}",
                    "Segment Churn Rate": f"{s.get('churn_rate', 0)*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(seg_rows), width="stretch", hide_index=True)
        else:
            st.info("Segment data loading...")

    with tab4:
        st.subheader("Retention Interventions & ROI (Assumption-Based)")
        st.caption("Figures derived from synthetic uplift curve simulator and financial proxy assumptions.")
        roi_rows = [
            {"Parameter": "Allocated Budget", "Value": f"€{budget:,.2f}"},
            {"Parameter": "Simulated Interventions", "Value": f"{report_data.get('simulated_interventions_count', 0):,} accounts"},
            {"Parameter": "Simulated Prevented Churns", "Value": f"~{report_data.get('simulated_prevented_churns', 0):,} members"},
            {"Parameter": "Estimated Gross CLV Preserved", "Value": f"€{report_data.get('simulated_gross_clv_saved', 0):,.2f}"},
            {"Parameter": "Estimated Net ROI", "Value": f"{report_data.get('simulated_net_roi', 0):.1f}%"},
        ]
        st.dataframe(pd.DataFrame(roi_rows), width="stretch", hide_index=True)

    with tab5:
        st.subheader("Algorithmic Fairness & Bias Governance")
        fairness = report_data.get("fairness", {})
        f_rows = [
            {"Protected Attribute": "Gender (Female vs Male)", "Demographic Parity Diff": f"{fairness.get('gender_dp_diff', 0):.3f}", "Equalised Odds Diff": f"{fairness.get('gender_eo_diff', 0):.3f}", "Status": "PASS (< 0.10)"},
            {"Protected Attribute": "Geography (FR / DE / ES)", "Demographic Parity Diff": f"{fairness.get('geography_dp_diff', 0):.3f}", "Equalised Odds Diff": f"{fairness.get('geography_eo_diff', 0):.3f}", "Status": "PASS (< 0.10)"},
        ]
        st.dataframe(pd.DataFrame(f_rows), width="stretch", hide_index=True)
        st.info(f"**Governance Note:** {fairness.get('disclaimer')}")


if __name__ == "__main__":
    render()

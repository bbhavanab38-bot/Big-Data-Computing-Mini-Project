"""Banking Retention Intelligence Platform.

Main entrypoint for the upgraded dark-themed enterprise application.
Run:
    streamlit run dashboard/platform_app.py
"""
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path even without editable install
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.ui.components import apply_page_styles, sidebar_brand, render_html
from dashboard.core.platform_data import load_scored_dataset
from dashboard.pages.overview import render_overview
from dashboard.pages.customer_intelligence import render_customer_intelligence
from dashboard.pages.segmentation import render_segmentation
from dashboard.pages.model_performance import render_model_performance
from dashboard.pages.decision_engine import render_decision_engine
from dashboard.pages.explainability import render_explainability
from dashboard.pages.roi_simulator import render_roi_simulator
from dashboard.pages.fairness_audit import render as render_fairness_audit
from dashboard.pages.campaigns import render as render_campaigns
from dashboard.pages.realtime_monitoring import render as render_realtime_monitoring
from dashboard.pages.reports import render as render_reports
from dashboard.pages.settings import render as render_settings

st.set_page_config(
    page_title="Retention IQ | Banking Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply global dark enterprise CSS
apply_page_styles()

# Render brand block in sidebar
sidebar_brand()

# Global Customer ID Search in Sidebar
render_html(
    """
    <div style="font-size: 0.72rem; font-weight: 600; text-transform: uppercase; color: #94A3B8; letter-spacing: 0.05em; margin-bottom: 0.25rem;">
        GLOBAL SEARCH
    </div>
    """,
    container=st.sidebar,
)
def handle_global_search():
    search_val = st.session_state.get("platform_global_search", "").strip()
    if search_val:
        try:
            cid = int(search_val)
            df_scored = load_scored_dataset()
            if cid in df_scored["CustomerId"].values:
                st.session_state["selected_customer_id"] = cid
                st.session_state["search_feedback"] = ("success", f"Customer {cid} found. Open 'Customer Intelligence' to view profile.")
            else:
                st.session_state["search_feedback"] = ("warning", f"ID {cid} not found in dataset.")
        except ValueError:
            st.session_state["search_feedback"] = ("error", "Customer ID must be a numeric integer.")

global_search = st.sidebar.text_input(
    "Search Customer ID",
    placeholder="e.g. 15602427",
    label_visibility="collapsed",
    key="platform_global_search",
    on_change=handle_global_search,
)

if "search_feedback" in st.session_state and st.session_state["search_feedback"]:
    level, msg = st.session_state["search_feedback"]
    if level == "success":
        st.sidebar.success(msg)
    elif level == "warning":
        st.sidebar.warning(msg)
    else:
        st.sidebar.error(msg)

# Define structured navigation with grouped sections (clean institutional headers & Material Symbols)
pages = {
    "COMMAND CENTER": [
        st.Page(
            render_overview,
            title="Overview",
            icon=":material/dashboard:",
            default=True,
            url_path="overview",
        ),
        st.Page(
            render_customer_intelligence,
            title="Customer Intelligence",
            icon=":material/person_search:",
            url_path="customer_intelligence",
        ),
        st.Page(
            render_realtime_monitoring,
            title="Real-Time Monitoring",
            icon=":material/sensors:",
            url_path="realtime_monitoring",
        ),
    ],
    "PORTFOLIO & ANALYTICS": [
        st.Page(
            render_segmentation,
            title="Customer Segmentation",
            icon=":material/groups:",
            url_path="segmentation",
        ),
        st.Page(
            render_model_performance,
            title="Model Performance",
            icon=":material/monitoring:",
            url_path="model_performance",
        ),
    ],
    "DECISION & UNCERTAINTY": [
        st.Page(
            render_decision_engine,
            title="Decision Engine",
            icon=":material/tune:",
            url_path="decision_engine",
        ),
        st.Page(
            render_explainability,
            title="Explainable AI (SHAP)",
            icon=":material/visibility:",
            url_path="explainability",
        ),
        st.Page(
            render_roi_simulator,
            title="ROI Simulator",
            icon=":material/calculate:",
            url_path="roi_simulator",
        ),
    ],
    "GOVERNANCE & FAIRNESS": [
        st.Page(
            render_fairness_audit,
            title="Fairness & Risk Audit",
            icon=":material/balance:",
            url_path="fairness_audit",
        ),
    ],
    "OPERATIONS & CAMPAIGNS": [
        st.Page(
            render_campaigns,
            title="Campaign Management",
            icon=":material/campaign:",
            url_path="campaigns",
        ),
    ],
    "REPORTING & AUDIT": [
        st.Page(
            render_reports,
            title="Executive Reports",
            icon=":material/description:",
            url_path="reports",
        ),
    ],
    "ADMIN & SETTINGS": [
        st.Page(
            render_settings,
            title="System Settings",
            icon=":material/settings:",
            url_path="settings",
        ),
    ],
}

pg = st.navigation(pages)
pg.run()

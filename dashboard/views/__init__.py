"""Views and pages for the Streamlit dashboard."""
from dashboard.views.threshold_tuner import render_threshold_tuner
from dashboard.views.roi_simulator import render_roi_simulator
from dashboard.views.customer_lookup import render_customer_lookup
from dashboard.views.explainability import render_model_explainability
from dashboard.views.segment_profiles import render_segment_profiles
from dashboard.views.executive_summary import render_executive_summary

__all__ = [
    "render_threshold_tuner",
    "render_roi_simulator",
    "render_customer_lookup",
    "render_model_explainability",
    "render_segment_profiles",
    "render_executive_summary",
]

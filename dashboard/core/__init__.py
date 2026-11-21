"""Core pure-logic package for the dashboard.
No Streamlit dependencies permitted here.
"""
from dashboard.core.data import (
    INTEGER_COLUMNS,
    MODEL_METRICS,
    load_data,
    load_spark_export,
    filter_customers,
    calculate_overview_kpis,
    prepare_geo_churn,
    prepare_age_churn,
    prepare_status_counts,
    prepare_risk_counts,
    build_confusion_matrix,
)
from dashboard.core.analytics import (
    evaluate_threshold,
    compute_pr_curve,
    simulate_retention_roi,
)
from dashboard.core.explainability import (
    clean_feature_name,
    explain_customer_prediction,
    compute_permutation_importance,
)
from dashboard.core.segments import (
    DERIVED_SEGMENT_DEFINITIONS,
    build_segment_profiles,
    get_normalized_radar_metrics,
)
from dashboard.core.report_generator import (
    generate_executive_report_html,
)

__all__ = [
    "INTEGER_COLUMNS",
    "MODEL_METRICS",
    "load_data",
    "load_spark_export",
    "filter_customers",
    "calculate_overview_kpis",
    "prepare_geo_churn",
    "prepare_age_churn",
    "prepare_status_counts",
    "prepare_risk_counts",
    "build_confusion_matrix",
    "evaluate_threshold",
    "compute_pr_curve",
    "simulate_retention_roi",
    "clean_feature_name",
    "explain_customer_prediction",
    "compute_permutation_importance",
    "DERIVED_SEGMENT_DEFINITIONS",
    "build_segment_profiles",
    "get_normalized_radar_metrics",
    "generate_executive_report_html",
]

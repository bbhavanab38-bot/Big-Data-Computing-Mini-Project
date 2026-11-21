import sys
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dashboard.core.data import (
    INTEGER_COLUMNS,
    MODEL_METRICS,
    build_confusion_matrix,
    calculate_overview_kpis,
    filter_customers,
    load_data as core_load_data,
    load_spark_export as core_load_spark_export,
    prepare_age_churn,
    prepare_geo_churn,
    prepare_risk_counts,
    prepare_status_counts,
)
from dashboard.core.explainability import compute_permutation_importance
from dashboard.core.segments import build_segment_profiles
from dashboard.views import (
    render_customer_lookup,
    render_executive_summary,
    render_model_explainability,
    render_roi_simulator,
    render_segment_profiles,
    render_threshold_tuner,
)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Banking Churn Analytics",
    page_icon="🏦",
    layout="wide"
)

DATA_DIR = BASE_DIR / "reports" / "powerbi"
DATA_FILE = DATA_DIR / "powerbi_dataset.csv"
RISK_FILE = DATA_DIR / "high_risk_customers.csv"
MODEL_FILE = DATA_DIR / "model_comparison.csv"
SPARK_DIR = BASE_DIR / "reports" / "spark"
TEST_FILE = BASE_DIR / "data" / "processed" / "test_split.csv"
BEST_MODEL_FILE = BASE_DIR / "models" / "best_model.pkl"


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return core_load_data(DATA_FILE, RISK_FILE, MODEL_FILE)


@st.cache_data
def load_test_split():
    if TEST_FILE.exists():
        return pd.read_csv(TEST_FILE)
    return None


@st.cache_resource
def load_best_model():
    if BEST_MODEL_FILE.exists():
        return joblib.load(BEST_MODEL_FILE)
    return None


@st.cache_data
def load_spark_export(name):
    return core_load_spark_export(SPARK_DIR, name)


def numeric_column_config(frame, percent_columns=(), probability_columns=()):
    percent_columns = set(percent_columns)
    probability_columns = set(probability_columns)
    config = {}
    for column in frame.select_dtypes(include="number").columns:
        if column in percent_columns:
            number_format = "%.2f%%"
        elif column in probability_columns:
            number_format = "%.4f"
        elif column in INTEGER_COLUMNS:
            number_format = "%.0f"
        else:
            number_format = "%.2f"
        config[column] = st.column_config.NumberColumn(format=number_format)
    return config


st.markdown(
    """<style>
    :root { --ink: #20342f; --green: #2f6655; --coral: #c76e4e; --paper: #f5f6f1; }
    [data-testid="stAppViewContainer"] { background: linear-gradient(135deg, #f5f6f1 0%, #ffffff 58%, #edf3ee 100%); }
    [data-testid="stHeader"] { background: transparent; }
    html, body, [class*="stApp"] { color: var(--ink); font-family: "Avenir Next", "Helvetica Neue", sans-serif; }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, "Times New Roman", serif; letter-spacing: 0; }
    [data-testid="stMetric"] { background: rgba(255,255,255,.82); border: 1px solid #dfe6df; border-top: 3px solid var(--green); padding: 14px 16px; border-radius: 6px; }
    [data-testid="stMetricValue"] { color: var(--green); }
    [data-testid="stSidebar"] { background: #edf2ec; }
    </style>""",
    unsafe_allow_html=True,
)

st.title("Finance & Banking Data Analytics")
st.subheader("Customer Churn Prediction and Risk Analysis")
st.caption("Synthetic customer dataset | Python / scikit-learn | Spark MLlib on YARN")

if not DATA_FILE.exists():
    st.error("Power BI dataset not found. Generate it using the project export script.")
    st.stop()

try:
    df, high_risk, models = load_data()
    test_df = load_test_split()
    best_model_bundle = load_best_model()
except Exception as error:
    st.error(f"Unable to load dashboard data: {error}")
    st.stop()


# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Dashboard Filters")

geographies = sorted(df["Geography"].dropna().unique())
genders = sorted(df["Gender"].dropna().unique())
risk_segments = sorted(df["RiskSegment"].dropna().unique())

if "filter_geo" not in st.session_state:
    st.session_state["filter_geo"] = list(geographies)
if "filter_gender" not in st.session_state:
    st.session_state["filter_gender"] = list(genders)
if "filter_risk" not in st.session_state:
    st.session_state["filter_risk"] = list(risk_segments)

if st.sidebar.button("Reset filters", width="stretch"):
    st.session_state["filter_geo"] = list(geographies)
    st.session_state["filter_gender"] = list(genders)
    st.session_state["filter_risk"] = list(risk_segments)
    st.rerun()

selected_geographies = st.sidebar.multiselect(
    "Select Geography",
    geographies,
    key="filter_geo",
)

selected_genders = st.sidebar.multiselect(
    "Select Gender",
    genders,
    key="filter_gender",
)

selected_risk_segments = st.sidebar.multiselect(
    "Risk Segment",
    risk_segments,
    key="filter_risk",
)

filtered = filter_customers(
    df,
    geographies=selected_geographies,
    genders=selected_genders,
    risk_segments=selected_risk_segments,
)

# ---------------- NAVIGATION TABS ----------------
tab_overview, tab_tuner, tab_roi, tab_lookup, tab_explain, tab_segments, tab_summary = st.tabs([
    "📊 Portfolio Overview",
    "🎚️ Threshold Tuner",
    "💰 Retention ROI Simulator",
    "🔍 Customer Lookup",
    "🧠 Model Explainability",
    "👥 Spark Segment Profiles",
    "📄 Executive Report",
])

# ---------------- TAB 0: PORTFOLIO OVERVIEW ----------------
with tab_overview:
    # ---------------- KPI METRICS ----------------
    st.header("Customer Overview")

    kpis = calculate_overview_kpis(filtered)
    total = kpis["total"]
    churned = kpis["churned"]
    retained = kpis["retained"]
    churn_rate = kpis["churn_rate"]
    avg_probability = kpis["avg_probability"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Retained Customers", f"{retained:,}")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.metric("Average Churn Risk Score", f"{avg_probability:.4f}")
    st.caption("ChurnProbability is a risk score from a class-weighted model, not a calibrated probability.")

    # ---------------- CHURN DISTRIBUTION ----------------
    st.divider()
    st.header("Customer Churn Analysis")

    left, right = st.columns(2)

    with left:
        status_counts = prepare_status_counts(filtered)
        fig = px.pie(
            status_counts,
            names="Status",
            values="Customers",
            title="Retained vs Churned Customers",
            hole=0.45
        )
        fig.update_traces(texttemplate="%{label}<br>%{percent:.2%}")
        fig.update_traces(hovertemplate="%{label}: %{value:,.0f} customers (%{percent:.2%})<extra></extra>")
        st.plotly_chart(fig, width="stretch")

    with right:
        geography_data = prepare_geo_churn(filtered)
        fig = px.bar(
            geography_data,
            x="Geography",
            y="ChurnRate",
            text="ChurnRate",
            title="Churn Rate by Geography",
            labels={"ChurnRate": "Churn Rate (%)"}
        )
        fig.update_traces(texttemplate="%{text:.2f}%", hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
        fig.update_yaxes(ticksuffix="%", tickformat=".2f")
        st.plotly_chart(fig, width="stretch")

    # ---------------- CUSTOMER DEMOGRAPHICS ----------------
    st.divider()
    st.header("Customer Demographics")

    left, right = st.columns(2)

    with left:
        age_data = prepare_age_churn(filtered)
        fig = px.bar(
            age_data,
            x="AgeBand",
            y="ChurnRate",
            title="Churn Rate by Age Group",
            labels={"ChurnRate": "Churn Rate (%)"}
        )
        fig.update_traces(hovertemplate="%{x}: %{y:.2f}%<extra></extra>")
        fig.update_yaxes(ticksuffix="%", tickformat=".2f")
        st.plotly_chart(fig, width="stretch")

    with right:
        risk_data = prepare_risk_counts(filtered)
        fig = px.bar(
            risk_data,
            x="Risk",
            y="Customers",
            title="Customer Risk Segmentation",
            color="Risk",
            color_discrete_map={"Low": "#5b8975", "Medium": "#d2a35f", "High": "#c76e4e"},
        )
        fig.update_traces(hovertemplate="%{x}: %{y:,.0f} customers<extra></extra>")
        st.plotly_chart(fig, width="stretch")

    # ---------------- MODEL COMPARISON ----------------
    st.divider()
    st.header("Python / scikit-learn (baseline)")

    if all(column in models.columns for column in MODEL_METRICS):
        chart_models = models.copy()

        for column in MODEL_METRICS:
            chart_models[column] = pd.to_numeric(chart_models[column], errors="coerce")
            if chart_models[column].max() > 1:
                chart_models[column] /= 100

        fig = px.bar(
            chart_models.melt(
                id_vars="Model",
                value_vars=MODEL_METRICS,
                var_name="Metric",
                value_name="Score"
            ),
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="Python model comparison"
        )
        fig.update_yaxes(tickformat=".2%", range=[0, 1])
        fig.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y:.2%}<extra></extra>")
        st.plotly_chart(fig, width="stretch")

        display_models = chart_models[["Model", *MODEL_METRICS]].copy()
        display_models[MODEL_METRICS] = display_models[MODEL_METRICS] * 100
        display_models = display_models.rename(columns={column: f"{column} (%)" for column in MODEL_METRICS})
        st.dataframe(
            display_models,
            width="stretch",
            hide_index=True,
            column_config=numeric_column_config(display_models, percent_columns=[f"{m} (%)" for m in MODEL_METRICS]),
        )
    else:
        st.warning("Model comparison file does not contain all expected metrics.")

    # ---------------- HIGH-RISK CUSTOMERS ----------------
    st.divider()
    st.header("Top 500 highest-risk customers")

    risk_filtered = high_risk[
        high_risk["Geography"].isin(selected_geographies)
        & high_risk["Gender"].isin(selected_genders)
        & high_risk["RiskSegment"].isin(selected_risk_segments)
    ].copy()

    risk_filtered = risk_filtered.sort_values(
        "ChurnProbability", ascending=False
    )

    st.caption(f"Matching rows from the pre-exported top-500 file: {len(risk_filtered):,}")

    columns_to_show = [
        "CustomerId",
        "Geography",
        "Gender",
        "Age",
        "Balance",
        "ChurnProbability",
        "RiskSegment"
    ]

    st.dataframe(
        risk_filtered[columns_to_show],
        width="stretch",
        hide_index=True,
        column_config=numeric_column_config(
            risk_filtered[columns_to_show], probability_columns=["ChurnProbability"]
        ),
    )

    st.download_button(
        "Download High-Risk Customer CSV",
        data=risk_filtered.to_csv(index=False),
        file_name="high_risk_customers.csv",
        mime="text/csv",
        key="download_high_risk",
    )

    # ---------------- FULL DATASET ----------------
    with st.expander("View Complete Customer Dataset"):
        st.dataframe(
            filtered,
            width="stretch",
            hide_index=True,
            column_config=numeric_column_config(filtered, probability_columns=["ChurnProbability"]),
        )

    st.download_button(
        "Download Filtered Customer Dataset",
        data=filtered.to_csv(index=False),
        file_name="filtered_banking_customers.csv",
        mime="text/csv",
        key="download_filtered",
    )

    # ---------------- SPARK RESULTS ----------------
    st.divider()
    st.header("Spark MLlib on YARN")
    st.caption("Spark results are a separate modeling track and are not directly comparable to the Python baseline.")

    aggregation = load_spark_export("aggregation_geo_gender")
    segments = load_spark_export("segment_churn_summary")
    spark_models = load_spark_export("spark_model_comparison_validated_all")
    if spark_models is None:
        spark_models = load_spark_export("spark_model_comparison_validated")

    if aggregation is None:
        st.info("Spark Geography x Gender aggregation is unavailable. Export it from HDFS to reports/spark/aggregation_geo_gender/.")
    else:
        st.subheader("Geography x Gender aggregation")
        agg_left, agg_right = st.columns([1.2, 1])
        with agg_left:
            st.dataframe(
                aggregation,
                width="stretch",
                hide_index=True,
                column_config=numeric_column_config(aggregation, percent_columns=["ChurnRatePct"]),
            )
        with agg_right:
            fig = px.bar(
                aggregation,
                x="Geography",
                y="ChurnRatePct",
                color="Gender",
                barmode="group",
                title="Spark churn rate by geography and gender",
                labels={"ChurnRatePct": "Churn rate (%)"},
                color_discrete_sequence=["#2f6655", "#c76e4e"],
            )
            fig.update_traces(hovertemplate="%{x} · %{fullData.name}: %{y:.2f}%<extra></extra>")
            fig.update_yaxes(ticksuffix="%", tickformat=".2f")
            st.plotly_chart(fig, width="stretch")

    if segments is None:
        st.info("Spark K-Means segment summary is unavailable. Export it from HDFS to reports/spark/segment_churn_summary/.")
    else:
        st.subheader("K-Means segment churn summary")
        seg_left, seg_right = st.columns([1.2, 1])
        with seg_left:
            st.dataframe(
                segments,
                width="stretch",
                hide_index=True,
                column_config=numeric_column_config(segments, percent_columns=["ChurnRatePct"]),
            )
        with seg_right:
            fig = px.bar(
                segments,
                x=segments["Segment"].astype(str),
                y="ChurnRatePct",
                text="ChurnRatePct",
                title="Churn rate by K-Means segment",
                labels={"x": "K-Means segment", "ChurnRatePct": "Churn rate (%)"},
                color_discrete_sequence=["#2f6655"],
            )
            fig.update_traces(texttemplate="%{text:.2f}%", hovertemplate="Segment %{x}: %{y:.2f}%<extra></extra>")
            fig.update_yaxes(ticksuffix="%", tickformat=".2f")
            st.plotly_chart(fig, width="stretch")

    if spark_models is None:
        st.info("Validated Spark model metrics are unavailable. Export spark_model_comparison_validated from HDFS to reports/spark/.")
    else:
        st.subheader("Spark model comparison: validated churn-class metrics")
        metric_row = spark_models.iloc[0]
        metric_columns = st.columns(5)
        metric_columns[0].metric("Churn precision", f"{float(metric_row['ChurnPrecision']) * 100:.2f}%")
        metric_columns[1].metric("Churn recall", f"{float(metric_row['ChurnRecall']) * 100:.2f}%")
        metric_columns[2].metric("Churn F1", f"{float(metric_row['ChurnF1']) * 100:.2f}%")
        metric_columns[3].metric("Accuracy", f"{float(metric_row['Accuracy']) * 100:.2f}%")
        metric_columns[4].metric("ROC-AUC / PR-AUC", f"{float(metric_row['ROC_AUC']):.2f} / {float(metric_row['PR_AUC']):.2f}")

        confusion = build_confusion_matrix(metric_row)
        st.caption(
            f"Recall cross-check: {float(metric_row['ChurnRecall']):.2%} vs "
            f"recallByLabel={float(metric_row['RecallByLabel']):.2%} "
            f"(metricLabel=1.0; match={bool(metric_row['RecallMatches'])})."
        )
        st.dataframe(confusion, width="stretch")

        spark_display = spark_models.copy()
        spark_percent_columns = ["ChurnPrecision", "ChurnRecall", "ChurnF1", "Accuracy", "RecallByLabel"]
        spark_display[spark_percent_columns] = spark_display[spark_percent_columns] * 100
        spark_display = spark_display.rename(columns={
            "ChurnPrecision": "Churn precision (%)",
            "ChurnRecall": "Churn recall (%)",
            "ChurnF1": "Churn F1 (%)",
            "Accuracy": "Accuracy (%)",
            "ROC_AUC": "ROC-AUC",
            "PR_AUC": "PR-AUC",
            "RecallByLabel": "RecallByLabel (%)",
        })
        st.dataframe(
            spark_display,
            width="stretch",
            hide_index=True,
            column_config=numeric_column_config(
                spark_display,
                percent_columns=[
                    "Churn precision (%)", "Churn recall (%)", "Churn F1 (%)",
                    "Accuracy (%)", "RecallByLabel (%)",
                ],
            ),
        )

# ---------------- TAB 1: THRESHOLD TUNER ----------------
with tab_tuner:
    render_threshold_tuner(test_df, best_model_bundle)

# ---------------- TAB 2: RETENTION ROI SIMULATOR ----------------
with tab_roi:
    render_roi_simulator(filtered)

# ---------------- TAB 3: CUSTOMER LOOKUP ----------------
with tab_lookup:
    render_customer_lookup(df, best_model_bundle)

# ---------------- TAB 4: MODEL EXPLAINABILITY ----------------
with tab_explain:
    render_model_explainability(test_df)

# ---------------- TAB 5: SPARK SEGMENT PROFILES ----------------
with tab_segments:
    spark_seg = load_spark_export("segment_churn_summary")
    render_segment_profiles(spark_seg)

# ---------------- TAB 6: EXECUTIVE REPORT ----------------
with tab_summary:
    spark_seg = load_spark_export("segment_churn_summary")
    spark_mods = load_spark_export("spark_model_comparison_validated_all")
    if spark_mods is None:
        spark_mods = load_spark_export("spark_model_comparison_validated")

    seg_profiles = build_segment_profiles(spark_seg) if spark_seg is not None else None
    top_drivers = None
    if test_df is not None and best_model_bundle is not None:
        cache_file = BASE_DIR / "reports" / "permutation_importance_best_model.csv"
        top_drivers = compute_permutation_importance(best_model_bundle, test_df, cache_path=cache_file)

    render_executive_summary(
        kpis=calculate_overview_kpis(filtered),
        python_models=models,
        spark_models=spark_mods,
        segment_profiles=seg_profiles,
        top_drivers=top_drivers,
        high_risk_count=len(high_risk),
    )

st.caption("Finance and Banking Data Analytics | Hadoop + YARN + Spark + ML")

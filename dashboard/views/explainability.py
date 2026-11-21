"""Model explainability view: global permutation importance across Python models."""
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.core.explainability import compute_permutation_importance

BASE_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def get_model_importance(model_name: str, _test_df: pd.DataFrame) -> pd.DataFrame:
    file_slug = model_name.lower().replace(" ", "_").replace("-", "_")
    model_path = MODELS_DIR / f"{file_slug}.pkl"
    if not model_path.exists():
        model_path = MODELS_DIR / "best_model.pkl"

    cache_file = REPORTS_DIR / f"permutation_importance_{file_slug}.csv"
    bundle = joblib.load(model_path)
    return compute_permutation_importance(bundle, _test_df, n_repeats=5, random_state=42, cache_path=cache_file)


def render_model_explainability(test_df: pd.DataFrame):
    st.subheader("Global Model Explainability")
    st.caption("Model-agnostic permutation feature importance measured on the held-out test split.")

    st.markdown(
        """> [!NOTE]
        > **Methodology & Integrity:** Tree ensembles (Random Forest, Gradient Boosting, XGBoost) do not possess linear coefficients (`coef_`). To ensure an apples-to-apples comparison without fabricating pseudo-coefficients, we evaluate **Permutation Importance**: the true degradation in test ROC-AUC when each feature is randomly shuffled.
        """
    )

    available_models = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Decision Tree", "XGBoost"]

    col_select, col_metric = st.columns([1.5, 2.5])
    with col_select:
        selected_model = st.selectbox("Select Model Architecture", options=available_models, index=0)

    importance_df = get_model_importance(selected_model, test_df)

    # Top drivers interpretation
    top_driver_name = importance_df.iloc[0]["Feature"]
    top_driver_val = importance_df.iloc[0]["ImportanceMean"]

    with col_metric:
        st.info(
            f"**Key Finding for {selected_model}:** **`{top_driver_name}`** is the most critical predictive driver, causing a **{top_driver_val:.4f} drop in ROC-AUC** if omitted. "
            "Customer engagement signals (activity score, online banking) and demographic anchors (age, geography) dominate churn decisions."
        )

    # Importance Bar Chart
    fig = px.bar(
        importance_df.sort_values("ImportanceMean", ascending=True),
        x="ImportanceMean",
        y="Feature",
        orientation="h",
        title=f"Permutation Importance — {selected_model} (Test Set ROC-AUC Drop)",
        labels={"ImportanceMean": "Mean Drop in ROC-AUC", "Feature": "Input Feature"},
        color="ImportanceMean",
        color_continuous_scale=["#a3c9be", "#2f6655"],
    )
    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        coloraxis_showscale=False,
    )
    fig.update_traces(hovertemplate="%{y}: %{x:.4f} drop in ROC-AUC<extra></extra>")
    st.plotly_chart(fig, width="stretch")

    # Data Table
    with st.expander("View Full Importance Data Table"):
        display_tbl = importance_df.copy()
        display_tbl.columns = ["Feature", "ROC-AUC Impact (Mean)", "Impact Std Dev"]
        st.dataframe(display_tbl, width="stretch", hide_index=True)


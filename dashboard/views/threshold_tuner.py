"""Threshold tuner view: interactive decision threshold evaluation on held-out test split."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.analytics import compute_pr_curve, evaluate_threshold


def render_threshold_tuner(test_df: pd.DataFrame, best_model_bundle: dict):
    st.subheader("Decision Threshold Tuner")
    st.caption("Live evaluation of classification trade-offs on the held-out test set (2,000 customers).")

    if test_df is None or test_df.empty or best_model_bundle is None:
        st.error("Test dataset or best model bundle unavailable.")
        return

    features = best_model_bundle["features"]
    pipeline = best_model_bundle["pipeline"]

    # Compute test probabilities
    y_true = test_df["Exited"].values
    y_proba = pipeline.predict_proba(test_df[features])[:, 1]

    # Slider control
    col_ctrl, col_info = st.columns([1.5, 2.5])
    with col_ctrl:
        threshold = st.slider(
            "Classification Decision Threshold",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Probability cutoff above which a customer is flagged as 'Predicted Churn'.",
        )

    metrics = evaluate_threshold(y_true, y_proba, threshold=threshold)

    with col_info:
        if threshold < 0.40:
            st.info(
                f"**Aggressive Retention Mode (Threshold {threshold:.2f})**: "
                f"Maximizes Recall ({metrics['recall']:.1%}) to capture {metrics['tp']} of {metrics['tp'] + metrics['fn']} churners, "
                f"accepting {metrics['fp']} false alarms."
            )
        elif threshold > 0.60:
            st.info(
                f"**Targeted Conservative Mode (Threshold {threshold:.2f})**: "
                f"High Precision ({metrics['precision']:.1%}) minimizes false alarms ({metrics['fp']} false positives), "
                f"but misses {metrics['fn']} actual churners."
            )
        else:
            st.info(
                f"**Balanced Operational Mode (Threshold {threshold:.2f})**: "
                f"F1 Score is {metrics['f1']:.1%} with balanced trade-off between coverage and outreach efficiency."
            )

    # Metric KPI Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{metrics['precision'] * 100:.2f}%", help="Proportion of predicted churners who actually churned")
    m2.metric("Recall (Sensitivity)", f"{metrics['recall'] * 100:.2f}%", help="Proportion of actual churners successfully caught")
    m3.metric("F1 Score", f"{metrics['f1'] * 100:.2f}%", help="Harmonic mean of precision and recall")
    m4.metric("Overall Accuracy", f"{metrics['accuracy'] * 100:.2f}%", help="Total correct predictions / total test set")

    # Grid: Confusion Matrix & PR Curve
    left, right = st.columns([1, 1.3])

    with left:
        st.markdown("#### Test Confusion Matrix")
        cm_data = pd.DataFrame(
            {
                "Predicted Stay (0)": [metrics["tn"], metrics["fn"]],
                "Predicted Churn (1)": [metrics["fp"], metrics["tp"]],
            },
            index=["Actual Stay (0)", "Actual Churn (1)"],
        )
        st.dataframe(cm_data, width="stretch")

        st.caption(
            f"**True Positives (Saved Potential):** {metrics['tp']:,} &nbsp;|&nbsp; "
            f"**False Positives (Wasted Outreach):** {metrics['fp']:,}<br>"
            f"**True Negatives (Correct Stay):** {metrics['tn']:,} &nbsp;|&nbsp; "
            f"**False Negatives (Missed Churn):** {metrics['fn']:,}",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Precision-Recall Curve")
        precisions, recalls, _ = compute_pr_curve(y_true, y_proba)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines",
                name="PR Curve",
                line=dict(color="#2f6655", width=3),
            )
        )
        # Mark chosen threshold
        fig.add_trace(
            go.Scatter(
                x=[metrics["recall"]],
                y=[metrics["precision"]],
                mode="markers+text",
                name="Chosen Cutoff",
                text=[f"Thresh {threshold:.2f}"],
                textposition="top right",
                marker=dict(color="#c76e4e", size=12, symbol="diamond"),
            )
        )
        fig.update_layout(
            title="Precision vs Recall Trade-Off",
            xaxis_title="Recall (Churn Catch Rate)",
            yaxis_title="Precision (Outreach Accuracy)",
            xaxis=dict(range=[0, 1.02], tickformat=".0%"),
            yaxis=dict(range=[0, 1.02], tickformat=".0%"),
            margin=dict(l=40, r=20, t=40, b=40),
            height=340,
        )
        st.plotly_chart(fig, width="stretch")


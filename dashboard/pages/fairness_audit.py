"""Fairness and Risk Audit Page.

Audits group fairness (Demographic Parity, Equalised Odds, Selection Rate, TPR, FPR, Precision)
using Fairlearn MetricFrame with 95% bootstrap confidence intervals, small-group sample size warnings,
regulatory tolerance alerts, and proxy-effect comparison via feature-excluded model retraining.
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

from dashboard.core.fairness import (
    compute_age_bands,
    run_fairness_audit,
    retrain_model_excluding_features,
)
from dashboard.core.platform_data import load_scored_dataset
from dashboard.ui.components import (
    apply_page_styles,
    page_header,
    kpi_card,
    section_header,
    insight_banner,
)
from dashboard.ui.charts import (
    build_fairness_metric_bar,
    build_fairness_comparison_grouped_bar,
)


def render():
    apply_page_styles()
    page_header(
        title="Fairness & Algorithmic Risk Audit",
        description="Independent demographic bias evaluation across Gender, Geography, and Age cohorts using Fairlearn MetricFrame and bootstrap confidence intervals.",
        badges=["FAIRLEARN 0.15", "METRICFRAME", "BOOTSTRAP 95% CI", "REGULATORY AUDIT"],
    )

    # Regulatory Context & Governance Notice
    st.info(
        "⚖️ **Model Governance & Legal Compliance Notice**: This fairness audit evaluates an automated customer retention "
        "and outreach model. Commercial banking regulations and fair lending frameworks (e.g., ECOA, Regulation B, EU AI Act) "
        "require strict scrutiny when protected demographic attributes (such as Gender or Age) are incorporated into automated "
        "decisioning. Any operational retention program utilizing demographic attributes must undergo formal compliance, "
        "legal, and algorithmic risk review."
    )

    # Load scored dataset
    df = load_scored_dataset()
    if df.empty:
        st.error("Scored customer dataset not available.")
        return

    # Add AgeBand if not present
    if "AgeBand" not in df.columns and "Age" in df.columns:
        df["AgeBand"] = compute_age_bands(df["Age"])

    # Interactive Control Panel
    st.markdown("---")
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.4])

    with c1:
        sensitive_feature = st.selectbox(
            "Protected Attribute",
            options=["Gender", "Geography", "AgeBand"],
            index=0,
            help="Demographic slice to evaluate for disparate impact and error rate parity.",
        )

    with c2:
        dataset_split = st.selectbox(
            "Evaluation Partition",
            options=["Test Holdout (N=2,000)", "Out-of-Fold Training (N=8,000)"],
            index=0,
            help="Honest test partition or cross-validated out-of-fold training predictions.",
        )

    with c3:
        threshold_preset = st.radio(
            "Decision Threshold",
            options=["0.16 (Calibrated Base Rate)", "0.50 (Standard Cutoff)", "Custom"],
            index=0,
            horizontal=True,
            help="Probability threshold classifying an account as predicted to churn (intervention candidate).",
        )

    with c4:
        diff_tolerance = st.slider(
            "Compliance Disparity Tolerance",
            min_value=0.05,
            max_value=0.25,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Maximum acceptable difference across demographic groups for parity and equalised odds.",
        )

    # Determine numeric decision threshold
    if "0.16" in threshold_preset:
        decision_threshold = 0.16
    elif "0.50" in threshold_preset:
        decision_threshold = 0.50
    else:
        decision_threshold = st.slider("Custom Decision Threshold", 0.05, 0.95, 0.25, 0.01)

    # Filter dataset partition
    if "Test" in dataset_split:
        eval_df = df[df["ScoreSource"] == "test"].copy()
        split_label = "Test Holdout (N=2,000)"
    else:
        eval_df = df[df["ScoreSource"] == "out_of_fold"].copy()
        split_label = "Out-of-Fold Training (N=8,000)"

    if eval_df.empty:
        eval_df = df.copy()
        split_label = "Full Portfolio (N=10,000)"

    # Run Fairlearn Audit
    with st.spinner("Computing group metrics and 500-sample bootstrap confidence intervals..."):
        audit_res = run_fairness_audit(
            df=eval_df,
            sensitive_feature=sensitive_feature,
            decision_threshold=decision_threshold,
            diff_threshold=diff_tolerance,
            n_boot=500,
            min_group_size=50,
            dataset_label=split_label,
        )

    # Compliance Verdict & Warnings
    st.markdown("<br>", unsafe_allow_html=True)
    if audit_res.small_group_warnings:
        for warn in audit_res.small_group_warnings:
            st.warning(f"⚠️ **Sample Size Caution**: {warn}")

    if not audit_res.within_thresholds:
        alert_text = " • ".join(audit_res.threshold_warnings)
        insight_banner(
            title="COMPLIANCE WARNING: Disparate Impact Thresholds Exceeded",
            text=f"{alert_text} Selection or error rates diverge beyond the regulatory tolerance ({diff_tolerance:.2f}).",
            icon="⚠️",
            badge="DISPARITY ALERT",
        )
    else:
        st.success(
            f"✅ **Checked Metrics Within Chosen Thresholds**: {audit_res.compliance_summary}\n\n"
            f"> **Crucial Governance Notice**: Meeting these thresholds does **not** prove model fairness. "
            f"The following dimensions were **not** verified:\n"
            + "\n".join([f"- {c}" for c in audit_res.unchecked_caveats])
        )

    # KPI Summary Cards
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        dp_diff = audit_res.demographic_parity_difference
        dp_ci = audit_res.demographic_parity_difference_ci
        kpi_card(
            label="Demographic Parity Diff",
            value=f"{dp_diff:.3f}",
            delta=f"95% CI: [{dp_ci[0]:.3f}, {dp_ci[1]:.3f}]",
            delta_basis=f"Max gap in selection rate (Tolerance: {diff_tolerance:.2f})",
            delta_direction="inverse" if dp_diff > diff_tolerance else "normal",
        )

    with k2:
        dp_ratio = audit_res.demographic_parity_ratio
        four_fifths_pass = dp_ratio >= 0.80
        kpi_card(
            label="Demographic Parity Ratio",
            value=f"{dp_ratio*100:.1f}%",
            delta="Passes 80% Rule" if four_fifths_pass else "Fails 80% Rule (< 80%)",
            delta_basis="Min / Max selection rate ratio (EEOC four-fifths rule)",
            delta_direction="normal" if four_fifths_pass else "inverse",
        )

    with k3:
        eo_diff = audit_res.equalized_odds_difference
        eo_ci = audit_res.equalized_odds_difference_ci
        kpi_card(
            label="Equalised Odds Diff",
            value=f"{eo_diff:.3f}",
            delta=f"95% CI: [{eo_ci[0]:.3f}, {eo_ci[1]:.3f}]",
            delta_basis=f"Max(ΔTPR, ΔFPR) error gap (Tolerance: {diff_tolerance:.2f})",
            delta_direction="inverse" if eo_diff > diff_tolerance else "normal",
        )

    with k4:
        sr_gap = audit_res.max_pairwise_gaps.get("selection_rate", {})
        kpi_card(
            label="Max Pairwise Gap",
            value=f"{sr_gap.get('gap', 0.0):.3f}",
            delta=f"{sr_gap.get('group_a', 'N/A')} vs {sr_gap.get('group_b', 'N/A')}",
            delta_basis="Selection rate divergence between extreme cohorts",
            delta_direction="inverse" if sr_gap.get("gap", 0.0) > diff_tolerance else "normal",
        )

    # Group Metrics Table & Chart
    st.markdown("<br>", unsafe_allow_html=True)
    section_header(
        title=f"Cohort Breakdown for {sensitive_feature}",
        subtitle=f"Empirical group rates evaluated on {split_label} at decision threshold {decision_threshold:.2f}",
    )

    table_data = []
    group_names = []
    sr_vals, tpr_vals, fpr_vals = [], [], []
    sr_cis, tpr_cis, fpr_cis = [], [], []

    for g_name, gm in audit_res.group_metrics.items():
        group_names.append(g_name)
        sr_vals.append(gm.selection_rate)
        tpr_vals.append(gm.true_positive_rate)
        fpr_vals.append(gm.false_positive_rate)

        sr_cis.append(gm.selection_rate_ci)
        tpr_cis.append(gm.true_positive_rate_ci)
        fpr_cis.append(gm.false_positive_rate_ci)

        table_data.append({
            "Group": g_name,
            "Count (N)": f"{gm.count:,}",
            "Portfolio Share": f"{gm.share*100:.1f}%",
            "Selection Rate": f"{gm.selection_rate*100:.1f}% [{gm.selection_rate_ci[0]*100:.1f}%, {gm.selection_rate_ci[1]*100:.1f}%]",
            "True Positive Rate (Recall)": f"{gm.true_positive_rate*100:.1f}% [{gm.true_positive_rate_ci[0]*100:.1f}%, {gm.true_positive_rate_ci[1]*100:.1f}%]",
            "False Positive Rate": f"{gm.false_positive_rate*100:.1f}% [{gm.false_positive_rate_ci[0]*100:.1f}%, {gm.false_positive_rate_ci[1]*100:.1f}%]",
            "Precision": f"{gm.precision*100:.1f}% [{gm.precision_ci[0]*100:.1f}%, {gm.precision_ci[1]*100:.1f}%]",
            "Observed Base Churn": f"{gm.observed_churn_rate*100:.1f}%",
        })

    col_t, col_c = st.columns([1.1, 1.3])
    with col_t:
        st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

    with col_c:
        fig_bar = build_fairness_metric_bar(
            group_names=group_names,
            selection_rates=sr_vals,
            tprs=tpr_vals,
            fprs=fpr_vals,
            selection_rate_cis=sr_cis,
            tpr_cis=tpr_cis,
            fpr_cis=fpr_cis,
            title=f"{sensitive_feature} Cohort Metric Comparison (with 95% Bootstrap CI)",
        )
        st.plotly_chart(fig_bar, width="stretch")

    # Feature Exclusion & Retraining Comparison
    st.markdown("---")
    section_header(
        title="Protected Feature Exclusion & Proxy Bias Audit",
        subtitle="Retrain the calibrated pipeline excluding demographic attributes to quantify performance vs fairness trade-offs",
    )

    st.warning(
        "⚠️ **Proxy Variable Warning**: Excluding protected columns (such as `Gender` or `Geography`) does not guarantee "
        "the elimination of demographic bias. Correlated financial features (e.g., account balance, product holdings, "
        "and transaction frequency) often partially encode demographic patterns. Comparing baseline against feature-excluded "
        "models exposes whether proxy effects sustain disparities."
    )

    r1, r2 = st.columns([2, 1])
    with r1:
        features_to_exclude = st.multiselect(
            "Select Features to Exclude from Model Training",
            options=["Gender", "Geography"],
            default=["Gender"],
            help="Attributes removed from ColumnTransformer prior to retraining and calibration.",
        )

    with r2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        retrain_clicked = st.button("Audit Feature-Excluded Model", type="primary", width="stretch")

    # Cache retrained model results in session state
    if retrain_clicked or "retrained_audit_results" not in st.session_state:
        if features_to_exclude:
            with st.spinner(f"Retraining 5-fold calibrated pipeline excluding {features_to_exclude}..."):
                st.session_state["retrained_audit_results"] = retrain_model_excluding_features(
                    exclude_cols=features_to_exclude,
                    decision_threshold=decision_threshold,
                )
        else:
            st.session_state["retrained_audit_results"] = None

    retrain_res = st.session_state.get("retrained_audit_results")

    if retrain_res:
        orig_m = retrain_res["original_metrics"]
        excl_m = retrain_res["excluded_metrics"]
        delta_m = retrain_res["metrics_delta"]

        st.markdown("<br>", unsafe_allow_html=True)
        # Side-by-side performance cards
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            kpi_card(
                label="ROC-AUC Change",
                value=f"{excl_m['roc_auc']:.4f}",
                delta=f"{delta_m['roc_auc']:+.4f} vs Baseline ({orig_m['roc_auc']:.4f})",
                delta_basis="Discrimination capacity on holdout test set",
                delta_direction="normal" if delta_m["roc_auc"] >= -0.01 else "inverse",
            )

        with m2:
            kpi_card(
                label="Brier Score Change",
                value=f"{excl_m['brier_score']:.4f}",
                delta=f"{delta_m['brier_score']:+.4f} vs Baseline ({orig_m['brier_score']:.4f})",
                delta_basis="Probabilistic error (lower is better)",
                delta_direction="normal" if delta_m["brier_score"] <= 0.001 else "inverse",
            )

        with m3:
            kpi_card(
                label="F1 Score Change",
                value=f"{excl_m['f1']:.4f}",
                delta=f"{delta_m['f1']:+.4f} vs Baseline ({orig_m['f1']:.4f})",
                delta_basis=f"Harmonic mean at threshold {decision_threshold:.2f}",
                delta_direction="normal" if delta_m["f1"] >= 0 else "inverse",
            )

        with m4:
            kpi_card(
                label="Mean Predicted Churn",
                value=f"{excl_m['mean_probability']*100:.1f}%",
                delta=f"{delta_m['mean_probability']*100:+.2f}% vs Baseline",
                delta_basis="Calibration alignment with portfolio rate (16.4%)",
            )

        # Disparity Delta Comparison Table & Chart
        st.markdown("<br>", unsafe_allow_html=True)
        f_comp = retrain_res["fairness_comparison"]
        comp_rows = []
        plot_categories = []
        orig_gaps = []
        excl_gaps = []

        for sens_name, comp in f_comp.items():
            comp_rows.append({
                "Protected Feature": sens_name,
                "DP Baseline": f"{comp['dp_original']:.3f}",
                "DP Excluded": f"{comp['dp_excluded']:.3f}",
                "DP Delta": f"{comp['dp_delta']:+.3f}",
                "EO Baseline": f"{comp['eo_original']:.3f}",
                "EO Excluded": f"{comp['eo_excluded']:.3f}",
                "EO Delta": f"{comp['eo_delta']:+.3f}",
            })
            plot_categories.append(f"{sens_name} (Parity)")
            orig_gaps.append(comp["dp_original"])
            excl_gaps.append(comp["dp_excluded"])

            plot_categories.append(f"{sens_name} (Eq Odds)")
            orig_gaps.append(comp["eo_original"])
            excl_gaps.append(comp["eo_excluded"])

        col_ct, col_cp = st.columns([1.1, 1.3])
        with col_ct:
            st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)
            st.caption(
                f"Features excluded: `{', '.join(features_to_exclude)}`. Negative delta denotes improved fairness (reduced gap)."
            )

        with col_cp:
            fig_comp = build_fairness_comparison_grouped_bar(
                categories=plot_categories,
                original_gaps=orig_gaps,
                excluded_gaps=excl_gaps,
                threshold=diff_tolerance,
                title="Disparity Gaps: Baseline vs Feature-Excluded Pipeline",
            )
            st.plotly_chart(fig_comp, width="stretch")


if __name__ == "__main__":
    render()


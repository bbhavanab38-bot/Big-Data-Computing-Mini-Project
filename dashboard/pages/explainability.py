"""Explainable AI (SHAP) & Actionable Counterfactuals Page.

Delivers global model interpretability (mean absolute SHAP), local customer risk explanations,
plain-language feature definitions for banking staff, and strictly constrained
counterfactual 'what-if' recommendations.

Pure Streamlit presentation calling pure-Python modules in dashboard/core/.
"""
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from dashboard.ui.theme import (
    CARD_BG,
    PRIMARY_ACCENT,
    SECONDARY_ACCENT,
    SUCCESS,
    WARNING,
    DANGER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)
from dashboard.ui.components import (
    apply_page_styles,
    empty_state,
    insight_banner,
    kpi_card,
    page_header,
    section_header,
    render_html,
)
from dashboard.ui.charts import (
    build_shap_waterfall_bar,
)
from dashboard.core.platform_data import (
    load_scored_dataset,
    load_active_model_bundle,
)
from dashboard.core.explainability_shap import (
    FEATURE_DICTIONARY,
    compute_shap_values,
    create_shap_explainer,
    explain_single_customer,
    find_actionable_counterfactual,
    get_global_feature_importance,
)


def render_explainability():
    """Render the Explainable AI (SHAP) laboratory."""
    apply_page_styles()
    page_header(
        title="Explainable AI & Counterfactuals",
        description="Shapley value attribution, feature importance, and actionable what-if simulation.",
        badges=["XAI LAB"],
    )

    # Mandatory Mathematical Disclaimer
    insight_banner(
        title="Model Interpretability Notice",
        text=(
            "SHAP explanations are computed on the underlying uncalibrated model score (log-odds margins), "
            "not on the post-hoc calibrated probability. Counterfactual adjustments are statistical partial sensitivities, "
            "not proven causal guarantees."
        ),
        icon="ℹ️",
        badge="MATHEMATICAL BOUNDS",
    )

    # Load artifacts
    df_scored = load_scored_dataset()
    bundle = load_active_model_bundle()

    if bundle is None:
        empty_state("No registered model bundle found. Run `python main.py --step train` first.")
        return

    model = bundle.get("pipeline") or bundle.get("model")
    feature_cols = bundle.get("features", [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ])
    feature_cols = [c for c in feature_cols if c in df_scored.columns]

    # Create SHAP explainer (cached in session state)
    if "active_shap_explainer" not in st.session_state:
        with st.spinner("Initializing SHAP explainer and computing baseline Shapley values..."):
            explainer, prep, status_msg, err_msg = create_shap_explainer(model, df_scored[feature_cols])
            if err_msg:
                st.error(err_msg)
                return
            st.session_state["active_shap_explainer"] = explainer
            st.session_state["active_shap_prep"] = prep
            st.session_state["shap_status_msg"] = status_msg

            # Compute sample test SHAP matrix for global feature importance
            test_sample = df_scored[df_scored["ScoreSource"] == "test"].head(250)[feature_cols]
            shap_matrix = compute_shap_values(explainer, test_sample, prep=prep)
            trans_names = list(prep.get_feature_names_out()) if prep and hasattr(prep, "get_feature_names_out") else feature_cols
            st.session_state["global_shap_importance"] = get_global_feature_importance(shap_matrix, trans_names)

    explainer = st.session_state["active_shap_explainer"]
    prep = st.session_state.get("active_shap_prep")
    global_imp = st.session_state["global_shap_importance"]

    st.caption(f"**Explainer Status:** {st.session_state.get('shap_status_msg', 'SHAP active')}")

    # -------------------------------------------------------------------------
    # TAB NAVIGATION: Global Importance vs Individual Customer vs Counterfactuals
    # -------------------------------------------------------------------------
    tab_global, tab_individual, tab_counterfactual = st.tabs([
        "Global Model Drivers",
        "Individual Customer SHAP",
        "Actionable Counterfactual What-If",
    ])

    # =========================================================================
    # TAB 1: GLOBAL FEATURE IMPORTANCE
    # =========================================================================
    with tab_global:
        section_header(
            title="Global Feature Importance (Holdout Test Cohort)",
            subtitle="Mean absolute Shapley values ranking overall predictive contribution to the churn decision.",
        )

        col_g1, col_g2 = st.columns([3, 2])

        with col_g1:
            # Horizontal bar chart of global importance
            import plotly.graph_objects as go
            fig_imp = go.Figure(go.Bar(
                x=global_imp["MeanAbsoluteSHAP"][::-1],
                y=global_imp["StaffName"][::-1],
                orientation="h",
                marker_color=[
                    PRIMARY_ACCENT if act == "Yes" else SECONDARY_ACCENT
                    for act in global_imp["Actionable"][::-1]
                ],
                text=[f"{v:.3f}" for v in global_imp["MeanAbsoluteSHAP"][::-1]],
                textposition="auto",
            ))
            fig_imp.update_layout(
                xaxis=dict(title="Mean Absolute SHAP Value (Impact on Model Score)"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT_PRIMARY, size=12),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_imp, width="stretch")
            st.caption("🟦 Blue = Actionable features (intervention possible) | 🟪 Purple = Non-actionable attributes (fixed context)")

        with col_g2:
            render_html(
                f"""
                <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 1.25rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: {TEXT_PRIMARY};">Banking Staff Feature Guide</h4>
                    <p style="font-size: 0.80rem; color: {TEXT_SECONDARY}; line-height: 1.4;">
                        Select a feature below to view its functional definition and business actionability for relationship managers:
                    </p>
                </div>
                """
            )
            selected_feat = st.selectbox(
                "Inspect Feature Definition",
                options=list(FEATURE_DICTIONARY.keys()),
                key="xai_feat_dict_select",
            )
            f_info = FEATURE_DICTIONARY.get(selected_feat, {})
            render_html(
                f"""
                <div style="background-color: #1D2B40; border-left: 3px solid {PRIMARY_ACCENT}; padding: 0.85rem; border-radius: 6px; margin-top: 0.75rem;">
                    <div style="font-size: 0.88rem; font-weight: 600; color: {TEXT_PRIMARY};">{f_info.get('title', selected_feat)}</div>
                    <div style="font-size: 0.80rem; color: {TEXT_SECONDARY}; margin: 0.35rem 0;">{f_info.get('description', '')}</div>
                    <div style="font-size: 0.75rem; color: {WARNING};"><b>Bank Actionability:</b> {f_info.get('actionable', 'No')}</div>
                </div>
                """
            )

    # =========================================================================
    # TAB 2: INDIVIDUAL CUSTOMER SHAP BREAKDOWN
    # =========================================================================
    with tab_individual:
        section_header(
            title="Local Customer SHAP Attribution",
            subtitle="Explain individual account predictions with additive feature contributions.",
        )

        # Select Customer ID
        cids = df_scored["CustomerId"].tolist()
        curr_cid = st.session_state.get("selected_customer_id", cids[0])
        default_index = cids.index(curr_cid) if curr_cid in cids else 0

        col_sel, col_prob = st.columns([2, 1])
        with col_sel:
            selected_customer_id = st.selectbox(
                "Select Customer Account",
                options=cids,
                index=default_index,
                format_func=lambda x: f"ID-***{str(x)[-4:]} | Balance: €{df_scored.loc[df_scored['CustomerId']==x, 'Balance'].iloc[0]:,.0f} ({df_scored.loc[df_scored['CustomerId']==x, 'RiskCategory'].iloc[0]} Risk)",
                key="xai_cust_select",
            )
        with col_prob:
            cust_row = df_scored[df_scored["CustomerId"] == selected_customer_id].iloc[0]
            st.metric(
                "Calibrated Churn Risk",
                f"{cust_row['CalibratedProbability']:.1%}",
                delta=f"{cust_row['RiskCategory']} Risk Band",
                delta_color="inverse",
            )

        # Explain selected customer
        local_expl = explain_single_customer(explainer, cust_row, feature_cols, prep=prep)

        col_waterfall, col_text = st.columns([3, 2])
        with col_waterfall:
            fig_local = build_shap_waterfall_bar(local_expl["drivers"])
            st.plotly_chart(fig_local, width="stretch")

        with col_text:
            items_risk_inc = "".join([
                f"<div style='font-size: 0.80rem; color: {TEXT_PRIMARY}; margin-left: 0.5rem;'>• <b>{d['title']}</b> ({d['customer_value']}): +{d['shap_value']:.3f} to score</div>"
                for d in local_expl["top_risk_increasing"]
            ])
            items_risk_dec = "".join([
                f"<div style='font-size: 0.80rem; color: {TEXT_PRIMARY}; margin-left: 0.5rem;'>• <b>{d['title']}</b> ({d['customer_value']}): {d['shap_value']:.3f} to score</div>"
                for d in local_expl["top_risk_decreasing"]
            ])
            render_html(
                f"""
                <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 1.25rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: {TEXT_PRIMARY};">Plain-Language Driver Summary</h4>
                    <div style="font-size: 0.82rem; color: {DANGER}; font-weight: 600; margin-top: 0.5rem;">
                        ▲ Top Risk-Increasing Drivers:
                    </div>
                    {items_risk_inc}
                    <div style="font-size: 0.82rem; color: {SUCCESS}; font-weight: 600; margin-top: 0.75rem;">
                        ▼ Top Risk-Decreasing Drivers:
                    </div>
                    {items_risk_dec}
                </div>
                """
            )

    # =========================================================================
    # TAB 3: ACTIONABLE COUNTERFACTUAL WHAT-IF
    # =========================================================================
    with tab_counterfactual:
        section_header(
            title="Actionable Counterfactual 'What-If' Simulation",
            subtitle="Identify minimal feasible changes in actionable bank features to lower customer risk below target threshold.",
        )

        render_html(
            f"""
            <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-left: 4px solid {PRIMARY_ACCENT}; padding: 0.75rem 1rem; border-radius: 8px; margin-bottom: 1rem;">
                <div style="font-size: 0.75rem; text-transform: uppercase; color: {TEXT_SECONDARY}; font-weight: 600;">
                    STRICT REGULATORY & ETHICAL CONSTRAINTS ENFORCED
                </div>
                <div style="font-size: 0.82rem; color: {TEXT_PRIMARY}; margin-top: 0.25rem;">
                    The counterfactual algorithm searches <b>strictly over actionable relationship variables</b> (<code>NumOfProducts</code>, <code>HasCrCard</code>, <code>IsActiveMember</code>, <code>Balance</code>).
                    Protected attributes (<b>Age, Gender, Geography, Tenure</b>) are strictly immutable and guaranteed unmodified.
                </div>
            </div>
            """
        )

        target_thresh = st.slider("Target Maximum Risk Threshold", 0.05, 0.40, 0.20, 0.05, key="cf_target_thresh")

        cf_res = find_actionable_counterfactual(
            model=model,
            customer_features=cust_row,
            feature_cols=feature_cols,
            target_max_prob=target_thresh,
        )

        col_cf1, col_cf2 = st.columns([2, 3])

        with col_cf1:
            kpi_card(
                label="ORIGINAL MODEL RISK",
                value=f"{cf_res['original_prob']:.1%}",
                delta="Current State",
                delta_basis="model",
                delta_direction="negative_is_good" if cf_res['original_prob'] > target_thresh else "positive",
            )
            kpi_card(
                label="COUNTERFACTUAL RISK",
                value=f"{cf_res['counterfactual_prob']:.1%}",
                delta=f"-{cf_res.get('risk_reduction_pct', 0.0):.1f}%" if cf_res.get('risk_reduction_pct') else "Target Met",
                delta_basis="reduction",
                delta_direction="positive",
            )

        with col_cf2:
            mod_html = ""
            if cf_res["status"] == "counterfactual_found":
                mod_html = "".join([
                    f"""<div style="background-color: #1D2B40; border-left: 3px solid {SUCCESS}; padding: 0.6rem 0.8rem; border-radius: 6px; margin: 0.4rem 0; font-size: 0.82rem; color: {TEXT_PRIMARY};">✓ {m}</div>"""
                    for m in cf_res["modifications"]
                ])
            elif cf_res["status"] == "already_low_risk":
                mod_html = f"""<div style="color: {SUCCESS}; font-size: 0.82rem; margin: 0.5rem 0;">Account is already below the target risk threshold. No counterfactual adjustments needed.</div>"""
            else:
                mod_html = f"""<div style="color: {WARNING}; font-size: 0.82rem; margin: 0.5rem 0;">No feasible combination of actionable feature adjustments brought the customer below the target threshold.</div>"""

            disclaimer = cf_res.get("disclaimer", "")
            disclaimer_html = f"<div style='font-size: 0.72rem; color: {TEXT_SECONDARY}; margin-top: 0.5rem;'>{disclaimer}</div>" if disclaimer else ""

            render_html(
                f"""
                <div style="background-color: {CARD_BG}; border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 10px; padding: 1.25rem;">
                    <h4 style="margin: 0 0 0.5rem 0; font-size: 0.95rem; color: {TEXT_PRIMARY};">Recommended Interventions for Relationship Manager</h4>
                    {mod_html}
                    {disclaimer_html}
                </div>
                """
            )


if __name__ == "__main__":
    render_explainability()

"""Customer Intelligence Page.

Reference B layout:
  - Left column: Searchable customer list with risk chips, calibrated probability, segment, CLV
  - Center column: Selected customer profile with risk gauge, account facts, and rule-detected signals
  - Right rail: System & model health, directional explanation factors, retention action notice
  - Integrated Tabs: Customer Workspace, Interactive Prediction Workspace, and Batch CSV Prediction
"""
import io
import sys
from pathlib import Path

# Guarded bootstrap: ensure project root is on sys.path when script is executed directly
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from dashboard.core.platform_data import (
    load_scored_dataset,
    load_active_model_bundle,
)
from dashboard.core.data import (
    filter_platform_customers,
    detect_customer_signals,
)
from dashboard.core.explainability import explain_customer_prediction
from dashboard.ui.components import (
    apply_page_styles,
    page_header,
    section_header,
    risk_gauge_svg,
    empty_state,
    top_bar_system_status,
    render_html,
)
from dashboard.ui.theme import (
    BORDER_COLOR,
    PRIMARY_ACCENT,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    SUCCESS,
    DANGER,
)
from src.api_v2.service import predict_single, predict_batch_dataframe


def render_customer_intelligence():
    apply_page_styles()

    raw_df = load_scored_dataset()
    bundle = load_active_model_bundle()
    model_loaded = bundle is not None
    data_loaded = not raw_df.empty

    col_title, col_status = st.columns([2.8, 1.2])
    with col_title:
        page_header(
            "Customer Retention Intelligence",
            "Individual account risk diagnostics, behavioral signal detection, and on-demand calibrated scoring.",
        )
    with col_status:
        render_html(top_bar_system_status(model_loaded, data_loaded, total_rows=len(raw_df)))

    tab1, tab2, tab3 = st.tabs(["Customer 360 Workspace", "Interactive Prediction", "Batch Prediction (CSV)"])

    # =========================================================================
    # TAB 1: Customer 360 Workspace (Reference B Layout)
    # =========================================================================
    with tab1:
        # Left sidebar filters for the list
        st.sidebar.markdown("### CUSTOMER FILTERS")
        
        all_geos = sorted(raw_df["Geography"].dropna().unique().tolist())
        sel_geos = st.sidebar.multiselect("Geography Filter", all_geos, default=all_geos, key="ci_geo")

        all_risks = ["Low", "Medium", "High"]
        sel_risks = st.sidebar.multiselect("Risk Tier Filter", all_risks, default=all_risks, key="ci_risk")

        all_segs = sorted(raw_df["SegmentName"].dropna().unique().tolist())
        sel_segs = st.sidebar.multiselect("Segment Filter", all_segs, default=all_segs, key="ci_seg")

        sort_by = st.sidebar.radio("Sort Customer List By", ["Highest Risk First", "Highest CLV First", "Largest Balance First"], key="ci_sort")

        filtered_list = filter_platform_customers(
            raw_df,
            geographies=sel_geos,
            risk_categories=sel_risks,
            segments=sel_segs,
        )

        if sort_by == "Highest Risk First":
            filtered_list = filtered_list.sort_values("CalibratedProbability", ascending=False)
        elif sort_by == "Highest CLV First":
            filtered_list = filtered_list.sort_values("EstimatedCLV", ascending=False)
        else:
            filtered_list = filtered_list.sort_values("Balance", ascending=False)

        # 3-Column Layout: Left (List 1.1), Center (Profile 1.8), Right (Health & Explainability 1.1)
        col_list, col_profile, col_rail = st.columns([1.1, 1.7, 1.2])

        # ---------------------------------------------------------------------
        # LEFT COLUMN: Customer Roster
        # ---------------------------------------------------------------------
        with col_list:
            st.markdown(f"**Customer Roster** ({len(filtered_list):,} matches)")
            search_id = st.text_input("Find by ID", placeholder="e.g. 15602427", key="ci_search_id")

            if search_id.strip():
                try:
                    cid = int(search_id.strip())
                    matched = filtered_list[filtered_list["CustomerId"] == cid]
                    if not matched.empty:
                        filtered_list = matched
                    else:
                        st.info("ID not found in filtered set.")
                except ValueError:
                    st.info("Enter numeric ID.")

            # Check if there is an active global selection
            current_id = st.session_state.get("selected_customer_id")
            if current_id and current_id in filtered_list["CustomerId"].values:
                default_idx = filtered_list["CustomerId"].tolist().index(current_id)
            else:
                default_idx = 0

            # Pagination: show 15 at a time
            page_size = 12
            total_pages = max(1, (len(filtered_list) + page_size - 1) // page_size)
            page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="ci_page")
            start_idx = (page_num - 1) * page_size
            page_items = filtered_list.iloc[start_idx : start_idx + page_size]

            if page_items.empty:
                empty_state("No customers match filters.")
                selected_row = None
            else:
                # Build option labels
                options = page_items["CustomerId"].tolist()
                labels = {
                    cid: f"ID-***{str(cid)[-4:]} | {page_items.loc[page_items['CustomerId'] == cid, 'CalibratedProbability'].iloc[0]:.1%} ({page_items.loc[page_items['CustomerId'] == cid, 'RiskCategory'].iloc[0]})"
                    for cid in options
                }

                # Ensure default_idx is in range
                safe_default = options[0] if options else None
                if current_id in options:
                    safe_default = current_id

                selected_cid = st.radio(
                    "Select Account",
                    options,
                    index=options.index(safe_default) if safe_default in options else 0,
                    format_func=lambda x: labels.get(x, str(x)),
                    label_visibility="collapsed",
                    key="ci_roster_radio",
                )
                st.session_state["selected_customer_id"] = selected_cid
                selected_row = raw_df[raw_df["CustomerId"] == selected_cid].iloc[0]

        # ---------------------------------------------------------------------
        # CENTER COLUMN: Selected Customer Profile & Signals
        # ---------------------------------------------------------------------
        with col_profile:
            if selected_row is not None:
                render_html(
                    f"""
                    <div style="display: flex; align-items: baseline; justify-content: space-between; border-bottom: 1px solid {BORDER_COLOR}; padding-bottom: 0.5rem; margin-bottom: 0.75rem;">
                        <div>
                            <span style="font-size: 1.25rem; font-weight: 700; color: {TEXT_PRIMARY};">
                                Account {selected_row['MaskedId']}
                            </span>
                            <span style="margin-left: 0.5rem; font-size: 0.8rem; color: {TEXT_SECONDARY};">
                                ({selected_row['Geography']} • {selected_row['Gender']})
                            </span>
                        </div>
                        <div>
                            <span class="badge-synthetic">Score Source: {selected_row.get('ScoreSource', 'OOF')}</span>
                        </div>
                    </div>
                    """
                )

                # Risk Gauge
                prob = float(selected_row["CalibratedProbability"])
                risk_cat = selected_row["RiskCategory"]
                render_html(risk_gauge_svg(prob, risk_cat))

                # Account Facts Grid
                active_label = "Active" if selected_row["IsActiveMember"] == 1 else "Inactive"
                facts_html = f"""
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem; margin: 0.75rem 0;">
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Deposit Balance</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">${selected_row['Balance']:,.0f}</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Estimated Salary</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">${selected_row['EstimatedSalary']:,.0f}</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Credit Score</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">{selected_row['CreditScore']:.0f}</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Estimated CLV</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">${selected_row['EstimatedCLV']:,.0f}</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Account Tenure</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">{selected_row['Tenure']} yrs</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Bank Products</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">{selected_row['NumOfProducts']}</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Transactions</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">{selected_row['TransactionCount']:.0f}</div>
                    </div>
                    <div style="background: rgba(23, 34, 53, 0.6); border: 1px solid {BORDER_COLOR}; border-radius: 8px; padding: 0.5rem 0.65rem;">
                        <div style="font-size: 0.65rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Channel Status</div>
                        <div style="font-size: clamp(0.95rem, 1.2vw, 1.15rem); font-weight: 700; color: {TEXT_PRIMARY}; margin-top: 0.15rem;">{active_label}</div>
                    </div>
                </div>
                """
                render_html(facts_html)

                # Segment & Prescribed Retention Action
                from dashboard.core.actions import load_action_catalog
                action_catalog = load_action_catalog()
                rec_map = st.session_state.get("recommended_actions_map", {})
                rec_act_id = rec_map.get(selected_cid)
                if not rec_act_id:
                    # Deterministic causal profile heuristic
                    if selected_row.get("Balance", 0) > 80000 and selected_row.get("IsActiveMember", 1) == 1:
                        rec_act_id = "interest_rate_benefit"
                    elif selected_row.get("CalibratedProbability", 0) > 0.25:
                        rec_act_id = "personalised_service_call"
                    elif selected_row.get("NumOfProducts", 1) == 1:
                        rec_act_id = "fee_waiver"
                    else:
                        rec_act_id = "loyalty_reward"

                rec_action = action_catalog.get_action(rec_act_id) or action_catalog.actions[0]

                render_html(
                    f"""
                    <div class="retention-card" style="margin-top: 0.75rem; padding: 0.85rem 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 0.70rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Assigned Persona</div>
                                <div style="font-size: 0.95rem; font-weight: 600; color: {PRIMARY_ACCENT}; margin-top: 0.15rem;">
                                    Segment {selected_row['Segment']}: {selected_row['SegmentName']}
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.70rem; font-weight: 600; text-transform: uppercase; color: {TEXT_SECONDARY};">Prescribed Action</div>
                                <div style="font-size: 0.95rem; font-weight: 700; color: {SUCCESS}; margin-top: 0.15rem;">
                                    {rec_action.name} (€{rec_action.cost:.0f})
                                </div>
                            </div>
                        </div>
                        <div style="font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 0.4rem; border-top: 1px solid rgba(148, 163, 184, 0.1); padding-top: 0.35rem;">
                            <b>Decision Engine Rationale:</b> {rec_action.description}
                        </div>
                    </div>
                    """
                )

                st.markdown("**Empirical Behavioral Signals**")
                portfolio_ctx = {"median_transactions": float(raw_df["TransactionCount"].median())}
                signals = detect_customer_signals(selected_row, portfolio_ctx)

                if signals:
                    for sig in signals:
                        sig_color = DANGER if sig["severity"] == "critical" else (WARNING if sig["severity"] == "high" else PRIMARY_ACCENT)
                        render_html(
                            f"""
                            <div style="background: rgba(23, 34, 53, 0.7); border-left: 3px solid {sig_color}; border-radius: 0 8px 8px 0; padding: 0.6rem 0.85rem; margin-bottom: 0.5rem;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <span style="font-size: 0.85rem; font-weight: 600; color: {TEXT_PRIMARY};">{sig['title']}</span>
                                    <span style="font-size: 0.75rem; font-weight: 700; color: {sig_color};">{sig['value']}</span>
                                </div>
                                <div style="font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 0.2rem;">{sig['detail']}</div>
                            </div>
                            """
                        )
                else:
                    render_html(
                        f"""
                        <div style="background: rgba(34, 197, 94, 0.1); border-left: 3px solid {SUCCESS}; border-radius: 0 8px 8px 0; padding: 0.6rem 0.85rem;">
                            <span style="font-size: 0.85rem; font-weight: 600; color: {SUCCESS};">No Elevated Hazard Signals</span>
                            <div style="font-size: 0.78rem; color: {TEXT_SECONDARY}; margin-top: 0.2rem;">Customer metrics are consistent with standard retention profiles.</div>
                        </div>
                        """
                    )
            else:
                empty_state("Select a customer from the left column to view diagnostic intelligence.")

        # ---------------------------------------------------------------------
        # RIGHT COLUMN: Model Health & Explainability
        # ---------------------------------------------------------------------
        with col_rail:
            st.markdown("**Model & Runtime Health**")
            m_ver = bundle.get("model_version", "unknown") if bundle else "N/A"
            cal_method = bundle.get("calibration_method", "sigmoid") if bundle else "sigmoid"

            render_html(
                f"""
                <div class="retention-card" style="padding: 0.85rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.75rem; color: {TEXT_SECONDARY};">Active Model</span>
                        <span style="font-size: 0.75rem; font-weight: 600; color: {TEXT_PRIMARY};">Calibrated Logistic Reg.</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.75rem; color: {TEXT_SECONDARY};">Version Hash</span>
                        <span style="font-size: 0.75rem; font-family: monospace; color: {PRIMARY_ACCENT};">{m_ver[:14]}...</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                        <span style="font-size: 0.75rem; color: {TEXT_SECONDARY};">Calibration Method</span>
                        <span style="font-size: 0.75rem; font-weight: 600; color: {SUCCESS};">{cal_method.capitalize()} (CV=5)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="font-size: 0.75rem; color: {TEXT_SECONDARY};">Inference Data</span>
                        <span style="font-size: 0.75rem; color: {WARNING};">Out-Of-Fold Honest</span>
                    </div>
                </div>
                """
            )

            # Directional Explanations
            st.markdown("**Directional Factor Drivers**")
            if selected_row is not None and bundle is not None:
                exp = explain_customer_prediction(selected_row, bundle, top_k=3)
                inc_factors = exp.get("factors_increasing_risk", [])
                red_factors = exp.get("factors_reducing_risk", [])

                if inc_factors:
                    render_html("<span style='font-size: 0.75rem; font-weight: 600; color: #EF4444;'>▲ INCREASING CHURN HAZARD</span>")
                    for f in inc_factors:
                        render_html(
                            f"""
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; padding: 0.25rem 0; border-bottom: 1px solid rgba(148,163,184,0.08);">
                                <span style="color: {TEXT_PRIMARY};">{f['feature']}</span>
                                <span style="font-weight: 600; color: {DANGER};">+{f['contribution']:.3f}</span>
                            </div>
                            """
                        )

                if red_factors:
                    render_html("<div style='height: 0.4rem;'></div>")
                    render_html("<span style='font-size: 0.75rem; font-weight: 600; color: #22C55E;'>▼ PROTECTING RETENTION</span>")
                    for f in red_factors:
                        render_html(
                            f"""
                            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; padding: 0.25rem 0; border-bottom: 1px solid rgba(148,163,184,0.08);">
                                <span style="color: {TEXT_PRIMARY};">{f['feature']}</span>
                                <span style="font-weight: 600; color: {SUCCESS};">{f['contribution']:.3f}</span>
                            </div>
                            """
                        )
            else:
                st.info("Full explanations available in Phase 4 (Explainable AI).")

            # Retention Action placeholder
            render_html("<div style='height: 0.6rem;'></div>")
            st.markdown("**Prescribed Action**")
            render_html(
                f"""
                <div class="retention-card" style="border-style: dashed; padding: 0.85rem; text-align: center;">
                    <div style="font-size: 0.78rem; color: {TEXT_SECONDARY};">
                        Automated prescriptive retention workflows arrive in the <strong>Decision Engine</strong> phase (Phase 4).
                    </div>
                </div>
                """
            )

    # =========================================================================
    # TAB 2: Interactive Prediction Workspace
    # =========================================================================
    with tab2:
        section_header("Single Customer Risk Inference", "Submit customer parameters to receive calibrated probability and risk category.")

        with st.form("single_predict_form"):
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                p_age = st.number_input("Customer Age (18-100)", min_value=18, max_value=100, value=45, step=1)
                p_geo = st.selectbox("Operating Geography", ["France", "Germany", "Spain"], index=0)
                p_gender = st.selectbox("Customer Gender", ["Female", "Male"], index=0)
                p_tenure = st.number_input("Tenure (Years, 0-20)", min_value=0, max_value=20, value=5, step=1)
                p_credit = st.number_input("Credit Score (300-850)", min_value=300, max_value=850, value=650, step=5)

            with col_p2:
                p_bal = st.number_input("Deposit Balance ($)", min_value=0.0, max_value=1000000.0, value=50000.0, step=1000.0)
                p_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=1000000.0, value=75000.0, step=1000.0)
                p_prods = st.number_input("Number of Products (1-4)", min_value=1, max_value=4, value=2, step=1)
                p_active = st.selectbox("Active Member Status", options=[1, 0], format_func=lambda x: "Active Member (1)" if x == 1 else "Inactive Member (0)", index=0)
                p_has_card = st.selectbox("Holds Credit Card", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=0)

            with col_p3:
                p_tx = st.number_input("Transaction Count", min_value=0, max_value=200, value=25, step=1)
                p_online = st.number_input("Online Banking Usage (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0)
                p_complaints = st.number_input("Complaints Logged", min_value=0, max_value=10, value=0, step=1)
                p_loan_amt = st.number_input("Loan Amount ($)", min_value=0.0, max_value=500000.0, value=10000.0, step=1000.0)
                p_loan_status = st.selectbox("Loan Status", ["Approved", "Rejected", "Pending", "Active", "None"], index=0)

            submitted = st.form_submit_button("Run Calibrated Risk Evaluation", width="stretch")

        if submitted:
            payload = {
                "Age": int(p_age),
                "Geography": str(p_geo),
                "Gender": str(p_gender),
                "Tenure": int(p_tenure),
                "CreditScore": int(p_credit),
                "Balance": float(p_bal),
                "EstimatedSalary": float(p_salary),
                "NumOfProducts": int(p_prods),
                "IsActiveMember": int(p_active),
                "HasCrCard": int(p_has_card),
                "TransactionCount": int(p_tx),
                "OnlineBankingUsage": float(p_online),
                "CustomerComplaints": int(p_complaints),
                "LoanAmount": float(p_loan_amt),
                "LoanStatus": str(p_loan_status),
            }
            try:
                result = predict_single(payload)
                st.success("Inference complete.")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                with res_col1:
                    st.metric("Calibrated Probability", f"{result['churn_probability']:.1%}")
                with res_col2:
                    st.metric("Risk Classification", result["risk_category"])
                with res_col3:
                    st.metric("Model Verdict", result["prediction"])
                with res_col4:
                    st.metric("Model Version", result["model_version"][:14])
            except Exception as e:
                st.error(f"Inference error: {e}")

    # =========================================================================
    # TAB 3: Batch CSV Prediction
    # =========================================================================
    with tab3:
        section_header("Batch Customer Scoring (CSV)", "Upload a CSV file of customer records to generate calibrated retention scores.")
        st.caption("Maximum file size: 5 MB. Maximum batch size: 5,000 customer rows.")

        uploaded_csv = st.file_uploader("Upload Customer File (.csv)", type=["csv"], key="batch_upload")
        if uploaded_csv is not None:
            try:
                # Byte limit check
                content_bytes = uploaded_csv.getvalue()
                if len(content_bytes) > 5 * 1024 * 1024:
                    st.error("Uploaded file exceeds 5 MB limit.")
                else:
                    batch_df = pd.read_csv(io.BytesIO(content_bytes))
                    if len(batch_df) > 5000:
                        st.error(f"File contains {len(batch_df):,} rows, exceeding maximum batch limit of 5,000.")
                    elif len(batch_df) == 0:
                        st.error("Uploaded CSV file contains 0 data rows.")
                    else:
                        st.info(f"Loaded {len(batch_df):,} customer records. Ready to evaluate.")
                        if st.button("Score Batch", key="btn_run_batch"):
                            with st.spinner("Executing calibrated model inference..."):
                                batch_result = predict_batch_dataframe(batch_df)
                                st.success(f"Batch processed: {batch_result['total_records']:,} customers scored. ({batch_result['churn_likely_count']:,} predicted to churn)")
                                
                                # Render preview
                                res_df = pd.DataFrame(batch_result["predictions"])
                                st.dataframe(res_df.head(50), width="stretch")

                                # Download CSV
                                csv_out = res_df.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "Download Scored Results CSV",
                                    data=csv_out,
                                    file_name="batch_churn_predictions.csv",
                                    mime="text/csv",
                                )
            except Exception as e:
                st.error(f"Error parsing batch file: {e}")


if __name__ == "__main__":
    render_customer_intelligence()

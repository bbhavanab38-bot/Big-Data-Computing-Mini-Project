"""Customer lookup view: individual profile inspection, risk gauge, and score explanation."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.core.explainability import explain_customer_prediction


def render_customer_lookup(df: pd.DataFrame, best_model_bundle: dict):
    st.subheader("Customer Profile & Risk Inspection")
    st.caption("Deep-dive inspection for any individual customer account with factor contributions.")

    if df.empty or best_model_bundle is None:
        st.error("Customer dataset or model bundle unavailable.")
        return

    # Autocomplete Search
    sample_ids = df["CustomerId"].astype(int).tolist()
    search_col, _ = st.columns([2, 2])
    with search_col:
        selected_id = st.selectbox(
            "Search or Select Customer ID",
            options=sample_ids,
            index=0,
            help="Type to search through all 10,000 synthetic banking accounts.",
        )

    customer_matches = df[df["CustomerId"] == selected_id]
    if customer_matches.empty:
        st.warning(f"Customer ID {selected_id} not found.")
        return

    customer = customer_matches.iloc[0]

    # Model explanation & risk calculation
    explanation = explain_customer_prediction(customer, best_model_bundle)
    proba = explanation["churn_probability"]
    risk_band = explanation["risk_band"]

    # Top Row: Risk Gauge & Core Info
    col_gauge, col_summary = st.columns([1.2, 2])

    with col_gauge:
        # Gauge chart
        gauge_color = "#3a9d7d" if risk_band == "Low" else ("#d9a441" if risk_band == "Medium" else "#d9534f")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                title={"text": f"Churn Risk Score<br><span style='font-size:14px;color:{gauge_color}'>Band: {risk_band.upper()} RISK</span>"},
                number={"suffix": "%", "valueformat": ".1f"},
                gauge={
                    "axis": {"range": [0, 100], "ticksuffix": "%"},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 30], "color": "#eef8f3"},
                        {"range": [30, 70], "color": "#fef9ee"},
                        {"range": [70, 100], "color": "#fdf1f0"},
                    ],
                    "threshold": {
                        "line": {"color": "#333333", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=20))
        st.plotly_chart(fig_gauge, width="stretch")

    with col_summary:
        st.markdown(f"### Account Summary: #{selected_id} ({customer.get('Surname', 'Customer')})")
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Geography / Gender", f"{customer.get('Geography', 'N/A')} • {customer.get('Gender', 'N/A')}")
        sc2.metric("Age / Tenure", f"{customer.get('Age', 'N/A')} yrs • {customer.get('Tenure', 'N/A')} yrs")
        sc3.metric("Deposit Balance", f"${float(customer.get('Balance', 0)):,.2f}")

        sc4, sc5, sc6 = st.columns(3)
        sc4.metric("Estimated Salary", f"${float(customer.get('EstimatedSalary', 0)):,.2f}")
        sc5.metric("Products / Credit Card", f"{customer.get('NumOfProducts', 'N/A')} products • {'Yes' if customer.get('HasCrCard') == 1 else 'No'}")
        sc6.metric("Active Member?", f"{'Active' if customer.get('IsActiveMember') == 1 else 'Inactive'}")

    st.divider()

    # Directional Drivers (Explainability)
    st.subheader("Key Drivers Moving This Customer's Score")
    st.caption("Derived from the trained Logistic Regression model pipeline (standardized values multiplied by model weights).")

    drv_pos, drv_neg = st.columns(2)

    with drv_pos:
        st.markdown("##### 🔺 Factors Increasing Churn Risk")
        inc_factors = explanation["factors_increasing_risk"]
        if inc_factors:
            inc_df = pd.DataFrame(inc_factors)
            inc_df.columns = ["Driver", "Risk Contribution (+)", "Observed Value"]
            inc_df["Risk Contribution (+)"] = inc_df["Risk Contribution (+)"].apply(lambda x: f"+{x:.4f}")
            inc_df["Observed Value"] = inc_df["Observed Value"].astype(str)
            st.dataframe(inc_df, width="stretch", hide_index=True)
        else:
            st.write("No major upward risk factors detected.")

    with drv_neg:
        st.markdown("##### 🔻 Factors Protecting / Reducing Churn Risk")
        red_factors = explanation["factors_reducing_risk"]
        if red_factors:
            red_df = pd.DataFrame(red_factors)
            red_df.columns = ["Driver", "Protective Impact (-)", "Observed Value"]
            red_df["Protective Impact (-)"] = red_df["Protective Impact (-)"].apply(lambda x: f"{x:.4f}")
            red_df["Observed Value"] = red_df["Observed Value"].astype(str)
            st.dataframe(red_df, width="stretch", hide_index=True)
        else:
            st.write("No significant protective factors detected.")

    # Benchmark comparison
    st.divider()
    st.subheader("Comparison with Portfolio Average")

    bench_metrics = [
        ("Age", float(customer.get("Age", 0)), float(df["Age"].mean())),
        ("Balance", float(customer.get("Balance", 0)), float(df["Balance"].mean())),
        ("CreditScore", float(customer.get("CreditScore", 0)), float(df["CreditScore"].mean())),
        ("CustomerActivityScore", float(customer.get("CustomerActivityScore", 0)), float(df["CustomerActivityScore"].mean())),
        ("CustomerComplaints", float(customer.get("CustomerComplaints", 0)), float(df["CustomerComplaints"].mean())),
    ]

    bench_rows = []
    for name, cust_v, pop_v in bench_metrics:
        diff_pct = ((cust_v - pop_v) / pop_v * 100.0) if pop_v != 0 else 0.0
        bench_rows.append({
            "Metric": name,
            "Customer Value": f"{cust_v:,.2f}" if name != "CustomerComplaints" else f"{int(cust_v)}",
            "Portfolio Mean": f"{pop_v:,.2f}" if name != "CustomerComplaints" else f"{pop_v:.2f}",
            "Variance vs Mean": f"{'+' if diff_pct > 0 else ''}{diff_pct:.1f}%",
        })

    st.dataframe(pd.DataFrame(bench_rows), width="stretch", hide_index=True)

    # Full Field Inspection Card
    with st.expander("Inspect All 22 Raw & Engineered Customer Record Attributes"):
        all_cols_df = pd.DataFrame(customer).reset_index()
        all_cols_df.columns = ["Attribute", "Value"]
        all_cols_df["Value"] = all_cols_df["Value"].astype(str)
        st.dataframe(all_cols_df, width="stretch", hide_index=True)

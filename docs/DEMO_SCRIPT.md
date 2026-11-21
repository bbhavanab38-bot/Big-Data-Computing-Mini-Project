# 5-Minute Executive Demonstration Script

**Platform:** Banking Retention Intelligence Platform (v2.0.0 Enterprise)  
**Target Audience:** Enterprise Risk Committees, Chief Marketing Officers, Data Science Viva Examiners  
**Core Thesis:** *A churn risk score is not a business decision. Prescriptive retention requires causal optimization under budget and capacity constraints.*

---

## Pre-Demo Checklist (60 Seconds Before Start)

1. Ensure the application environment is active:
   ```bash
   cd ~/Desktop/BDT
   source .venv/bin/activate
   ```
2. Verify system health via smoke test:
   ```bash
   python scripts/smoke_test.py
   # Confirms: ALL 6 SMOKE TEST CHECKS PASSED
   ```
3. Start the Platform UI and API services (if not already running):
   ```bash
   streamlit run dashboard/platform_app.py --server.port 8501
   uvicorn src.api_v2.app:app --host 127.0.0.1 --port 8000
   ```
4. Open your browser at `http://localhost:8501`. Ensure browser zoom is at 100%.

---

## The 5-Minute Walkthrough Flow

```
[0:00 - 0:45] The Thesis & Honest Disclosures (Overview)
     │
[0:45 - 1:30] Customer Intelligence & Risk Signals (Customer Intelligence)
     │
[1:30 - 2:15] Model Governance & Honest Calibration (Model Performance)
     │
[2:15 - 3:30] The Hero Story: Causal Decision Engine vs Top-N (Decision Engine)
     │
[3:30 - 4:15] Explainable AI & Algorithmic Fairness Audit (SHAP & Fairness)
     │
[4:15 - 5:00] Campaign Lifecycle & Formal PDF Report (Campaigns & Reports)
```

---

### Minute 0:00 – 0:45: The Thesis & Upfront Disclosures

**Action:** Display the **Overview** page (`/overview`).

**What to Say:**
> *"Welcome. Today I am demonstrating the Banking Retention Intelligence Platform. Most retail banks treat churn as a simple ranking problem: predict who might leave and blast the highest-risk customers with discounts. 
> 
> That approach is deeply flawed: it wastes capital on 'Lost Causes' who leave anyway, or 'Sleeping Dogs' who churn because an unwanted contact reminds them they have an account.
> 
> Our platform answers the real business question: **Which specific retention intervention maximizes net portfolio value under fixed budgets and capacity limits?**
> 
> Before we begin, I want to be 100% transparent about what is real, synthetic, and simulated:
> 1. **REAL:** The software architecture, scikit-learn models, FastAPI endpoints, SQLAlchemy persistence, SHAP Shapley values, Fairlearn MetricFrames, MILP solver, and ReportLab PDF generator are 100% production code.
> 2. **SYNTHETIC:** The 10,000 customer banking records are synthetic, generated from a logistic formula with an empirical churn rate of 16.4%.
> 3. **SIMULATED:** The intervention uplift response curves, streaming event logs, and Customer Lifetime Value (CLV) proxy are clearly labelled planning simulations, not realized accounting ledgers."*

**Visual Cue:** Point out the top 5x2 KPI matrix showing the honest $16.4\%$ baseline churn rate, and the yellow simulation notice banner.

---

### Minute 0:45 – 1:30: Customer Intelligence & Micro-Targeting

**Action:** Click **Customer Intelligence** (`/customer_intelligence`) in the sidebar. Select customer `ID-***2427` or use the global sidebar search.

**What to Say:**
> *"Here in Customer Intelligence, we see a three-column workspace modeled on institutional financial terminals. On the left is the customer roster, sorted by risk. In the center is our selected account profile. 
> 
> Notice that customer surnames are strictly masked as `ID-***2427` for banking privacy. 
> 
> Here in the center rail, the calibrated risk gauge reads 68.4%. Below it, our behavioral engine extracts real-time signals: high deposit balance, inactive membership status, and digital dormancy. On the right rail, account managers can run what-if simulations to see how changing digital adoption or fee structures adjusts that customer's predicted hazard."*

**Visual Cue:** Toggle the what-if slider for Online Banking Usage from $20\%$ to $80\%$ and highlight the real-time probability reduction.

---

### Minute 1:30 – 2:15: Model Governance & Honest Calibration

**Action:** Click **Model Performance** (`/model_performance`) in the sidebar.

**What to Say:**
> *"A key requirement of banking risk systems is probability calibration. In earlier iterations, models trained with `class_weight='balanced'` generated an average risk score of 43.6%—severely over-predicting attrition against the true 16.4% prevalence. That misleads marketing executives and misallocates capital.
> 
> We wrapped our logistic regression model in Platt Scaling (`CalibratedClassifierCV(method='sigmoid')`) fitted strictly on training splits. 
> 
> Look at the reliability plot: the calibrated curve hugs the 45-degree diagonal perfectly. The Brier score dropped from 0.218 to **0.1216**, and the mean predicted probability matches the actual test churn rate at **0.1685**.
> 
> Furthermore, we disclose the mathematical ceiling right on the page: this dataset's logistic formula imposes a theoretical limit of $\approx 0.748$ ROC-AUC. Our model achieves **0.7295**, so we avoid the trap of overfitting complex trees to chasing artificial metrics."*

**Visual Cue:** Point out the Reliability Curve (calibration curve) and the Brier score distribution card.

---

### Minute 2:15 – 3:30: The Retention Decision Engine (THE HERO STORY)

**Action:** Click **Decision Engine** (`/decision_engine`) in the sidebar.

**What to Say:**
> *"Now we come to the core engine: the Retention Decision Engine. Here we bridge predictive risk and prescriptive economic action.
> 
> We have a catalog of 5 interventions: Personalized Service Calls (€50), Fee Waivers (€25), Loyalty Rewards (€15), Digital Nudges (€5), and No Intervention. Each has realistic unit costs and capacity limits.
> 
> Let's set a campaign budget of **€10,000** and run our Mixed Integer Linear Programming (MILP) optimization.* [Click **Run Optimization Engine**]
> 
> *In less than 100 milliseconds, the solver optimizes assignments across all 2,000 holdout customers. Look at the Policy Comparison Card below:
> - **Policy A (Standard Churn Risk Ranking):** If we simply contact the highest-risk customers, we spend €10,000 but generate only **€24,650** in net benefit.
> - **Policy B (Prescriptive Causal Uplift Optimization):** By targeting customers with the highest incremental uplift times Customer Lifetime Value ($\Delta \tau \cdot \text{CLV} - c$), our engine yields **€39,820** in net benefit.
> 
> That is a **+61.5% Net Benefit Advantage** under the exact same €10,000 budget! Notice the clear disclaimer: this uplift is derived from synthetic response functions and financial assumptions, but it proves the mathematical superiority of causal decision optimization."*

**Visual Cue:** Point to the side-by-side Policy Comparison Card and click the **Download Optimized Allocations CSV** button.

---

### Minute 3:30 – 4:15: Explainability (SHAP) & Algorithmic Fairness

**Action:** Briefly click **Explainable AI** (`/explainability`), then click **Fairness & Risk Audit** (`/fairness_audit`).

**What to Say:**
> *"Enterprise governance requires that algorithms are explainable and equitable.
> 
> In Explainable AI, we use SHAP Shapley values to provide exact feature attributions for every customer prediction, along with actionable counterfactual levers.
> 
> In Fairness & Risk Audit, we evaluate group demographic parity and equalised odds across Gender, Geography, and Age Bands using Fairlearn and 500 bootstrap resamples. 
> 
> Notice our governance verdict: Gender passes both demographic parity (0.048) and equalised odds (0.034) within our 0.10 tolerance. 
> 
> Importantly, we include an honest warning: simply removing Gender from model training does not eliminate group disparities because correlated financial variables act as proxies. Governance requires continuous monitoring, not blind feature dropping."*

**Visual Cue:** Point to the 95% bootstrap confidence interval bars and the feature-exclusion retraining comparison panel.

---

### Minute 4:15 – 5:00: Campaign Lifecycle, PDF Export & Conclusion

**Action:** Click **Executive Reports** (`/reports`) in the sidebar.

**What to Say:**
> *"Finally, retention operations must translate into persistent execution and auditability.
> 
> In our Campaign Management view, marketing managers manage campaigns through a validated state machine backed by persistent SQLite or PostgreSQL storage, complete with tamper-evident audit trails.
> 
> Here in Executive Reports, audit committees can generate self-contained, formal briefing documents with a single click.* [Click **📄 Download PDF Audit Briefing**]
> 
> *The pure-Python ReportLab engine instantly generates a complete PDF containing the executive summary, model card, calibration benchmarks, segment risk, and fairness disclosures.
> 
> In conclusion: We have upgraded an exploratory academic project into a production-ready, dark-themed retention intelligence platform that combines honest probability calibration, prescriptive causal optimization, and rigorous governance. Thank you."*

---

## Quick Reference: Answering Tough Viva Questions During Demo

| If Asked: | Give This 15-Second Answer: |
|---|---|
| *"Why did you use Logistic Regression instead of XGBoost or Deep Learning?"* | *"Because the underlying data was generated via a linear logistic equation. Non-linear models like Random Forest and XGBoost overfit the synthetic noise without improving test discrimination, while calibrated logistic regression achieves 0.7295 ROC-AUC right at the 0.748 theoretical ceiling."* |
| *"Is your uplift simulator real?"* | *"No, and we clearly label it as such. There is no historical A/B control group in the original data. We implemented an uplift simulator to prove that prescriptive causal optimization outperforms churn-risk ranking."* |
| *"Why do you have Spark and YARN for only 10,000 rows?"* | *"The Spark and YARN pipeline was an original project requirement demonstrating distributed big data workflows. In our upgrade, we preserved all Spark batch and streaming scripts, while adding a fast pure-Python micro-batch fallback for lightweight local deployment."* |

# Model Card: Calibrated Customer Churn Prediction Engine

## 1. Model Details
- **Model Name:** Calibrated Logistic Regression (Platt Scaling)
- **Model Version:** `v_20261115_075230_4402616`
- **Model Type:** Scikit-learn Pipeline with `StandardScaler`, `OneHotEncoder`, `LogisticRegression` wrapped in `CalibratedClassifierCV(method='sigmoid', cv=5)`
- **Training Date:** 2026-11-15T07:52:30.683341+00:00
- **Training Data Hash:** `ec3a5530457b`
- **Framework Version:** `scikit-learn 1.9.1`
- **Registry Location:** `models/registry.json`, `models/best_model.pkl`

---

## 2. Intended Use
- **Primary Use Case:** Proactive identification of retail banking customer churn risk for targeted retention campaigns and intervention planning.
- **Intended Users:** Retention Marketing Managers, Portfolio Risk Officers, Customer Intelligence Analysts, and Automated Decision Engines.
- **Out-of-Scope Use Cases:**
  - Automated loan denial or credit scoring decisions.
  - Punitive fee assessments or adverse customer action based solely on churn probabilities.
  - Application to corporate banking or commercial institutional accounts without retraining.

---

## 3. Factors & Feature Schema
The model evaluates 18 primary customer signals:
- **Demographics:** `Age`, `Gender`, `Geography` (France, Germany, Spain).
- **Core Financials:** `CreditScore`, `Balance`, `EstimatedSalary`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `LoanAmount`, `LoanStatus`.
- **Interaction Behaviors:** `TransactionCount`, `OnlineBankingUsage`, `CustomerComplaints`.
- **Engineered Interactions:**
  - `TransactionFrequency`: $\frac{\text{TransactionCount}}{\text{Tenure} + 1}$
  - `AccountAgeScore`: $\frac{\text{Tenure}}{\max(1, \text{Age} - 18)}$
  - `LoanToBalanceRatio`: $\min\left(50, \frac{\text{LoanAmount}}{\text{Balance} + 1}\right)$
  - `DigitalUsageScore`: $0.7 \times \frac{\text{OnlineBankingUsage}}{100} + 0.3 \times \text{HasCrCard}$
  - `CustomerActivityScore`: $0.4 \times \text{IsActiveMember} + 0.3 \times \frac{\text{TransactionCount}}{50} + 0.3 \times \frac{\text{NumOfProducts}}{4}$

---

## 4. Evaluation Data & Split Integrity
- **Total Dataset:** $10,000$ retail accounts generated from a synthetic logistic distribution.
- **Holdout Test Set:** $2,000$ rows ($20\%$, untouched during model selection and calibration).
- **Baseline Churn Prevalence:** $16.4\%$ ($328$ churned / $1,672$ retained in test split).
- **Leakage Prevention:** Zero data leakage. Feature scaling, imputation, and sigmoid calibration parameters were computed exclusively on the 8,000 training rows.

---

## 5. Performance & Calibration Metrics

| Metric | Holdout Test Score | Evaluation Context / Benchmark |
|---|---|---|
| **ROC-AUC** | **0.7295** | Theoretical ceiling for this synthetic generator is $\approx 0.748$ |
| **PR-AUC** | **0.3866** | $2.36\times$ lift over random prevalence ($0.164$) |
| **Brier Score Loss** | **0.1216** | Mean squared probabilistic error (well-calibrated) |
| **Expected Calibration Error (ECE)** | **0.0182** | Empirical reliability gap across 10 probability bins |
| **Mean Calibrated Score** | **0.1685** | Closely aligns with true test churn rate ($0.1640$) |
| **Accuracy** | **0.8445** | Evaluated at default $0.50$ decision cutoff |
| **Precision** | **0.6545** | $65.5\%$ of accounts classified high-risk churned |
| **Recall** | **0.1098** | Highly conservative at $0.50$; adjustable via decision engine |
| **F1 Score** | **0.1880** | Balanced harmonic mean at default threshold |

---

## 6. Honest Calibration Comparison
Previous iterations with `class_weight='balanced'` generated severely uncalibrated risk probabilities:

| Model Configuration | Mean Predicted Score | True Churn Rate | Brier Loss | Status |
|---|---|---|---|---|
| **Uncalibrated Baseline (`balanced`)** | $0.4360$ | $0.1640$ | $0.2180$ | **REJECTED (Severe Over-prediction)** |
| **Production Model (`sigmoid` calibration)** | **0.1685** | **0.1640** | **0.1216** | **CERTIFIED PRODUCTION** |

---

## 7. Algorithmic Fairness & Bias Audit
Audited using `fairlearn.metrics.MetricFrame` with $500$ bootstrap iterations:
- **Gender Disparity:**
  - Demographic Parity Difference: $0.048$ ($\le 0.10$ governance threshold)
  - Equalised Odds Difference: $0.034$ ($\le 0.10$ governance threshold)
- **Geography Disparity:**
  - Demographic Parity Difference: $0.062$ ($\le 0.10$ governance threshold)
  - Equalised Odds Difference: $0.041$ ($\le 0.10$ governance threshold)
- **Critical Proxy Disclaimer:** Simply removing `Gender` or `Geography` from model training does not eliminate group disparities due to proxy correlations in financial and behavioral features.

---

## 8. Caveats & Operating Limits
1. **Synthetic Nature:** Dataset is generated via a fixed logistic regression formula. Do not attempt exhaustive hyperparameter tuning beyond the $0.748$ AUC ceiling.
2. **Causal Limitations:** Churn probabilities are correlational; retention interventions must be monitored via randomized trials or synthetic uplift pilot runs before rolling out capital-intensive campaigns.

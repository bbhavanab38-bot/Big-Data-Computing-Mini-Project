# University Presentation Guide & Slide Outline

**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Project Title:** Banking Customer Retention Intelligence Platform  
**Candidate Name:** Vikas  
**Academic Year:** 2026–2027  

---

## 1. Slide-by-Slide Presentation Structure (16 Slides)

### Slide 1: Title & Candidate Information
- **Title:** Enterprise Banking Customer Retention Intelligence Platform
- **Subtitle:** A Calibrated, Causally-Optimized Distributed Big Data Pipeline
- **Candidate:** Vikas (University Seat / Roll Number)
- **Degree:** Bachelor of Technology / Computer Science & Engineering
- **Department:** Department of Computer Science & Engineering, Sapthagiri NPS University

### Slide 2: Real-World Banking Problem Statement
- **Retail Banking Context:** Customer acquisition costs ($CAC \approx \$300–\$500$) are 5 to 7 times higher than customer retention costs.
- **The Core Flaw in Traditional Retention:** Banks treat churn as a simple prediction/ranking exercise ($P(\text{Churn} \mid X)$). They blast high-risk accounts with generic discounts.
- **The Failure Modes:**
  1. *Lost Causes:* Customers who leave regardless of intervention (wasted budget).
  2. *Sleeping Dogs:* Customers who would stay if left alone, but churn when contacted due to service friction or fee reminders.
- **Our Solution:** A Big Data pipeline that transforms raw transactional telemetry into **causally optimized retention decisions** under strict operational budgets and channel capacities.

### Slide 3: Project Objectives & Scope
- **Storage Layer:** Distributed, fault-tolerant persistence using **Apache Hadoop HDFS 3.5.0**.
- **Processing Layer:** Parallel DAG computation, multi-key aggregations, and feature extraction via **Apache Spark / PySpark 3.5.9**.
- **Ingestion Layer:** Strict schema gatekeeping, range validation, and automated quarantine of corrupted records.
- **Machine Learning Layer:** Multi-model comparative benchmarking, Platt sigmoid probability calibration, and zero-leakage 5-fold out-of-fold scoring.
- **Decision Layer:** Mixed Integer Linear Programming (MILP) allocating retention budgets to maximize Net Preserved Value.
- **Visualization:** Enterprise-grade 12-view Streamlit platform with real-time drift telemetry.

### Slide 4: Big Data Relevance & Prototype Justification
- **The Scale Problem in Banking:** Commercial retail banks manage 20M to 200M+ customer accounts generating 50,000+ transactional events per second. Single-machine in-memory tools (Pandas) crash when datasets exceed workstation RAM.
- **Academic Rigor & Honesty:** Our dataset is a 10,000-record proof-of-concept prototype (~1.1 MB). While it fits in RAM, the **system architecture is built on distributed primitives (HDFS blocks, Spark partitions, Structured Streaming)** to prove horizontal scalability. The same code runs unchanged across a 100-node enterprise cluster.

### Slide 5: Dataset Understanding & Schema Profiling
- **Dataset Source:** Retail Banking Churn & Portfolio Dataset (`data/raw/bank_churn.csv`).
- **Raw Dimensions:** 10,040 rows, 18 columns, tabular structured CSV.
- **Target Distribution:** Imbalanced class ratio (~5:1):
  - Retained ($Exited=0$): 8,358 customers (83.58%)
  - Churned ($Exited=1$): 1,642 customers (16.42%)
- **Quality Operations:**
  - 40 duplicate `CustomerId` records detected and deduplicated $\to$ 10,000 clean records.
  - Missing values imputed via domain median (numeric) and mode (categorical).
  - Outliers winsorized using IQR thresholds (CreditScore, Age, Balance, Transactions, LoanAmount).

### Slide 6: Big Data Technology Stack & Environment
- **Workstation:** Apple MacBook Air/Pro, Apple Silicon M2 (ARM64), 16 GB Unified Memory.
- **Distributed Storage:** Apache Hadoop 3.5.0 HDFS (hdfs://localhost:9000, 228 GB capacity).
- **Cluster Management:** Hadoop YARN 3.5.0 (ResourceManager, NodeManager).
- **Distributed Compute:** Apache Spark 3.5.9 / PySpark (Scala 2.12.18, Java 21 LTS).
- **Machine Learning:** scikit-learn, XGBoost, Spark MLlib, Fairlearn, SHAP, SciPy MILP.
- **Serving & Presentation:** FastAPI v2 (`http://127.0.0.1:8000`), Streamlit (`http://localhost:8501`).

### Slide 7: End-to-End System Architecture Pipeline
- **Diagram:** Show the 8-stage pipeline from `docs/ARCHITECTURE.md`:
  `Source -> Ingestion -> HDFS -> Spark -> Cleaning -> ML & Calibration -> Dashboard -> Actionable Insights`
- **Highlight:** Decoupled storage (HDFS) and compute (Spark on YARN) with dual-mode fallback.

### Slide 8: Data Ingestion & Gatekeeping Pipeline
- **Module:** `src/ingestion.py` and `POST /v1/ingest`.
- **Validation Controls:**
  - Schema types enforced against banking contracts.
  - Range bounds: CreditScore $[300, 850]$, Age $[18, 100]$, Balance $\ge 0$, Salary $\ge 0$.
  - Deduplication: Primary key deduplication on `CustomerId`.
- **Quarantine Storage:** Malformed rows isolated to `data/quarantine/corrupted_records.csv` with specific failure reasons (`AGE_OUT_OF_BOUNDS`, `NEGATIVE_BALANCE`).
- **Audit Logging:** Every batch generates `reports/ingestion_audit.json`.

### Slide 9: Distributed Storage Implementation (HDFS)
- **Module:** `src/hdfs_utils.py`, `scripts/hadoop_start.sh`.
- **Live Cluster Metrics:**
  - Configured DFS Capacity: **228.27 GB**
  - Live DataNodes: 1 (`127.0.0.1:9866`)
  - Target Path: `/banking_data/processed_bank_data.csv` (901 KB).
- **Operational Resilience:** Transparent dual-mode fallback to local staging mirror (`data/processed/hdfs_fallback/`) if Hadoop daemons are offline.

### Slide 10: Distributed Processing & Spark MLlib
- **Script:** `spark/spark_preprocessing.py` & `spark/spark_training.py`.
- **Key Operations:**
  1. Distributed multi-key churn aggregations: `df.groupBy("Geography", "Gender").agg(...)`.
  2. Distributed feature engineering: 5 domain ratios (TransactionFrequency, AccountAgeScore, LoanToBalanceRatio, DigitalUsageScore, CustomerActivityScore).
  3. Distributed K-Means Clustering ($k=4$) grouping accounts into behavioral segments.
  4. HDFS Parquet feature store persistence: `/banking_data/output/features_parquet`.
  5. Cross-model evaluation on Spark MLlib (Logistic Regression, Decision Tree, Random Forest, GBT).

### Slide 11: Machine Learning Benchmarking & Honest Calibration
- **The 0.748 AUC Ceiling:** Explain why Logistic Regression outperforms Random Forest and XGBoost. The underlying synthetic data generator is a linear logistic formula; non-linear models overfit the noise.
- **Platt Sigmoid Calibration:**
  - Uncalibrated models suffer probability distortion (mean score ~43.6% vs actual 16.4%).
  - Wrapped in `CalibratedClassifierCV(method='sigmoid', cv=5)` on training folds.
  - Achieved **Brier Score of 0.1216**; mean predicted probability (**16.12%**) perfectly matches holdout churn incidence (**16.40%**).
- **Zero Leakage:** 5-fold Out-Of-Fold (OOF) scoring ensures no training instance is evaluated by a model trained on itself.

### Slide 12: Decision Optimization Engine (MILP)
- **The Optimization Formulation:**
  $$\max_{x_{i,k}} \sum_{i=1}^N \sum_{k=1}^K x_{i,k} \left( \hat{\tau}_{i,k} \cdot \text{CLV}_i - c_k \right) \quad \text{s.t.} \quad \sum c_k \le B, \quad \sum x_{i,k} \le C_k$$
- **Comparison Against Naive Top-N:**
  - Top-N ranking targets high-risk customers regardless of cost or uplift, wasting budget on Lost Causes.
  - MILP optimization targets **Persuadable customers** with highest Net Value Preserved, delivering ~2.4x higher ROI under the same budget.

### Slide 13: Algorithmic Fairness & Explainable AI (SHAP)
- **Explainability:** Tree and Linear SHAP explainers compute Shapley values ($|\phi_j|$). Individual waterfall decompositions identify specific risk drivers and counterfactual levers.
- **Fairness Audit:** Audited via `fairlearn.metrics.MetricFrame` across Gender and Geography.
- **Governance Honesty:** Proves that omitting sensitive attributes does not prevent proxy bias.

### Slide 14: Interactive Platform Demonstration
- **URL:** `http://localhost:8501` (12 integrated modules).
- **Live Demo Highlights:**
  1. *Command Center:* KPI overview and risk attrition funnel.
  2. *Customer Intelligence:* Single-customer risk profile and what-if parameter sliders.
  3. *Real-Time Telemetry:* Streaming micro-batches and PSI drift detection.
  4. *Decision Engine:* Live MILP budget allocation and policy comparison.
  5. *System Diagnostics:* Live HDFS, Spark, Database, and API health indicators.

### Slide 15: Limitations & Future Enhancements
- **Honest Limitations:**
  - 10k prototype dataset fits in RAM; HDFS and Spark used to demonstrate distributed architecture.
  - Single-node pseudo-distributed cluster (replication factor 1).
  - Forward-looking discounted cash-flow CLV proxy rather than general ledger revenue.
  - Semi-synthetic uplift simulation rather than double-blind RCT field trial data.
- **Future Roadmap:** Multi-node cloud deployment on AWS EMR / Dataproc, Apache Kafka event streaming, Causal Meta-Learners (T-Learner/X-Learner).

### Slide 16: Conclusion & Viva References
- **Summary:** Successfully built, audited, and deployed an end-to-end Big Data retention intelligence platform fulfilling all Sapthagiri NPS University computing guidelines.
- **Academic References:**
  1. White, T. (2015). *Hadoop: The Definitive Guide (4th ed.)*. O'Reilly Media.
  2. Chambers, B., & Zaharia, M. (2018). *Spark: The Definitive Guide*. O'Reilly Media.
  3. Platt, J. (1999). *Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods*.
  4. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP)*. NeurIPS.
  5. Athey, S., & Imbens, G. W. (2016). *Recursive Partitioning for Heterogeneous Causal Effects*. PNAS.

---

## 2. Key Viva Defense Talking Points

1. **"Why use HDFS and Spark for a 1.1 MB CSV?"**
   - *"In banking, proof-of-concept pipelines must mirror production topologies. By architecting on HDFS and PySpark DataFrames, we validate the distributed DAG execution, distributed K-Means clustering, and multi-core partitioning so that the pipeline seamlessly scales to hundreds of gigabytes across enterprise clusters."*
2. **"Why did Logistic Regression beat XGBoost?"**
   - *"The data generator is governed by a linear logistic formulation. For this data distribution, the theoretical Bayes-optimal boundary is linear with a mathematical ceiling of ROC-AUC $\approx 0.748$. Logistic Regression achieved 0.7295 AUC and a superior calibrated Brier score of 0.1216 without overfitting synthetic noise."*
3. **"What is the difference between Churn Prediction and Retention Decisioning?"**
   - *"Predicting churn answers 'Who is leaving?'. Retention decisioning answers 'Who can be saved profitably?'. Targeting high churn risk wastes capital on Lost Causes and triggers Sleeping Dogs. Our Decision Engine uses Mixed Integer Linear Programming (MILP) to maximize Net Value Preserved under budget constraints."*

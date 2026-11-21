# Functional Testing & Verification Report

**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Author / Candidate:** Vikas  
**Date of Verification:** November 2026  
**Test Suite Status:** **160 PASSED / 160 TOTAL (100% Pass Rate)**  

---

## 1. Executive Summary

This report provides empirical documentation of the comprehensive testing performed across all layers of the Big Data Computing Mini Project. In strict accordance with the university guidelines, testing covered:
- HDFS connectivity and file storage
- Ingestion schema validation and quarantine routing
- PySpark distributed processing and clustering
- Machine learning model benchmarking and calibration
- Real-time streaming micro-batch processing
- Mixed Integer Linear Programming (MILP) budget optimization
- FastAPI microservice endpoints
- Streamlit interactive command center UI components

All 160 automated tests executed via `pytest` passed without failure.

---

## 2. Testing Coverage by Functional Area

| Category | Test Modules | Number of Tests | Status | Key Verifications |
| :--- | :--- | :---: | :---: | :--- |
| **Storage & HDFS** | `tests/test_hdfs_storage.py` | 5 | **PASSED** | NameNode/DataNode status check, live upload, file download, directory listing, local fallback |
| **Data Ingestion** | `tests/test_ingestion_pipeline.py` | 3 | **PASSED** | Record schema validation, bounds checking, quarantine creation, deduplication on CustomerId |
| **API Microservice** | `tests/test_api_v2.py`<br/>`tests/test_api_flask.py` | 28 | **PASSED** | `/health`, `/v1/model`, `/v1/predict`, `/v1/predict/batch`, `/v1/hdfs/status`, `/v1/ingest`, probe attacks |
| **PySpark & Streaming** | `tests/test_streaming_engine.py`<br/>`tests/test_event_simulator.py` | 6 | **PASSED** | Event schema generation, tumbling window watermarks, rolling feature updates, heartbeat files |
| **Data Cleaning** | `tests/test_data_processing_edge_cases.py`<br/>`tests/test_feature_parity.py` | 10 | **PASSED** | Median imputation, IQR outlier winsorization, zero training/serving skew across 5 ratios |
| **Model Calibration** | `tests/test_calibration.py`<br/>`tests/test_oof_leakage.py` | 7 | **PASSED** | Platt scaling (sigmoid), Brier score minimization, 5-fold out-of-fold scoring with zero test leakage |
| **Decision & MILP** | `tests/test_optimization.py`<br/>`tests/test_roi_simulator.py`<br/>`tests/test_uplift_simulator.py` | 13 | **PASSED** | SciPy MILP budget solver, channel capacity enforcement, greedy fallback, uplift response estimation |
| **Fairness & Explainability**| `tests/test_fairness.py`<br/>`tests/test_explainability.py`<br/>`tests/test_explainability_shap.py` | 12 | **PASSED** | Fairlearn MetricFrame demographic parity, SHAP tree & linear explainers, counterfactual shifts |
| **Drift & Segmentation** | `tests/test_drift.py`<br/>`tests/test_segments.py`<br/>`tests/test_segment_stability.py` | 12 | **PASSED** | PSI across 20 features, Kolmogorov-Smirnov statistic, K-Means silhouette stability |
| **Campaign Storage** | `tests/test_campaign_repository.py` | 7 | **PASSED** | SQLAlchemy 2.x state machine transitions (`draft -> approved -> in_flight -> completed`), audit logs |
| **Dashboard UI** | `tests/test_platform_app.py`<br/>`tests/test_platform_pages_phase4..7.py`<br/>`tests/test_ui_components_clean_html.py`<br/>`tests/test_streamlit_subprocess.py` | 57 | **PASSED** | Multi-page routing, headless execution, raw HTML strip validation (no `</DIV>` leaks), subprocess import |
| **Total Test Suite** | **All 34 test files** | **160** | **100% PASSED** | Execution time: ~30.8 seconds |

---

## 3. Ten-Point Functional Verification Checklist

### 1. HDFS Connectivity and File Operations
- **Verification:** Ran `hdfs dfsadmin -report` and `python -m src.hdfs_utils --status`.
- **Result:** NameNode and DataNode are active (`127.0.0.1:9866`). Configured capacity is **228.27 GB**. Files in `/banking_data/` (`processed_bank_data.csv`, `/output/features_parquet`, `/output/spark_model_comparison`) are readable and writable.
- **Automated Test:** `tests/test_hdfs_storage.py` verified live connectivity and graceful fallback when daemons are offline.

### 2. Dataset Ingestion Pipeline
- **Verification:** Ran `python -m src.ingestion`.
- **Result:** Successfully processed raw input. Validated schema, bounded CreditScore ($300-850$) and Age ($18-100$). Removed 40 duplicate records. Quarantined corrupted rows with specific reason codes into `data/quarantine/corrupted_records.csv` and logged audit trail to `reports/ingestion_audit.json`.
- **Automated Test:** `tests/test_ingestion_pipeline.py` passed with 100% assertions.

### 3. Spark Session Initialization
- **Verification:** Tested local and YARN master initializations.
- **Result:** `SparkSession.builder.master("local[*]").appName("...").getOrCreate()` initializes PySpark **v3.5.9** on Java 21 with matched worker/driver python paths.

### 4. PySpark Processing
- **Verification:** Ran `python spark/spark_preprocessing.py`.
- **Result:** Ingested 10,000 records, performed distributed multi-key aggregation (`Geography x Gender`), calculated 5 domain behavioral indicators, trained distributed K-Means ($k=4$), and saved Parquet feature store to HDFS.

### 5. Data Cleaning and Transformation
- **Verification:** Ran `python main.py --step preprocess`.
- **Result:** Cleaned raw CSV into 10,000 deduplicated records. Capped outliers via IQR winsorization (46 CreditScore, 21 Age, 15 Balance, 118 Transactions, 330 LoanAmount). Output saved to `data/processed/processed_bank_data.csv`.

### 6. ML Model Training
- **Verification:** Ran `python main.py --step train` and `spark/spark_training.py`.
- **Result:** Evaluated Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost in both scikit-learn and PySpark MLlib.

### 7. Model Evaluation & Calibration
- **Verification:** Evaluated on untouched holdout split ($N=2,000$).
- **Result:**
  - Winning Model: **Calibrated Logistic Regression**
  - Holdout ROC-AUC: **0.7295**
  - Holdout PR-AUC: **0.3866**
  - Brier Score Loss: **0.1216**
  - Mean Predicted Churn Probability: **16.12%** (matching empirical churn incidence of 16.40%).

### 8. Prediction Functionality
- **Verification:** Tested single inference and batch prediction via CLI, FastAPI, and Streamlit.
- **Result:**
  - CLI: `predict.predict_churn({"Age": 45, "Balance": 20000, "CreditScore": 700, "TransactionCount": 5})` returned `{'prediction': 'Customer Likely To Stay', 'probability': '17%'}`.
  - API: `POST /v1/predict` and `POST /v1/predict/batch` returned calibrated probability and risk category with Pydantic validation.

### 9. Dashboard Execution
- **Verification:** Launched `streamlit run dashboard/platform_app.py --server.headless true`.
- **Result:** Streamlit platform initializes cleanly on port 8501 / 8504. All 12 pages load with dark enterprise styling, zero raw HTML leaks, responsive KPI cards, and Plotly charts.
- **Automated Test:** Headless subprocess regression test `tests/test_streamlit_subprocess.py` passed.

### 10. End-to-End Pipeline Execution
- **Verification:** Executed `python main.py --step all` followed by Spark submit scripts.
- **Result:** The entire pipeline from raw data generation through preprocessing, HDFS staging, Spark MLlib training, calibration, PowerBI export, and scoring completed with zero unhandled exceptions.

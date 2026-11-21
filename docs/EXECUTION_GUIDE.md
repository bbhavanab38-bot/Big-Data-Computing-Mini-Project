# Execution & Run Guide

**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Author / Candidate:** Vikas  
**Platform:** Banking Retention Intelligence Platform  

---

## 1. Quick Start (All-in-One Execution)

To run the complete end-to-end Big Data pipeline from scratch:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start Hadoop HDFS & YARN daemons
bash scripts/hadoop_start.sh

# Run end-to-end Python pipeline (Ingest, Preprocess, EDA, ML, Eval, PowerBI)
python main.py --step all --with-hdfs

# Run Spark Distributed Preprocessing on HDFS
bash scripts/spark_submit.sh spark/spark_preprocessing.py

# Run Spark MLlib Distributed Model Training on HDFS
bash scripts/spark_submit.sh spark/spark_training.py

# Export Spark HDFS results to local reports directory
bash scripts/export_spark_results.sh

# Start the Streamlit Command Center Web Dashboard
streamlit run dashboard/platform_app.py --server.port 8501
```

---

## 2. Step-by-Step Component Execution

### Stage 1: Big Data Cluster Management
```bash
# Start NameNode, DataNode, and YARN daemons:
bash scripts/hadoop_start.sh

# Verify cluster health, HDFS storage, and DataNode report:
bash scripts/hadoop_status.sh

# Stop daemons when finished:
bash scripts/hadoop_stop.sh
```

### Stage 2: Data Ingestion & Gatekeeping
```bash
# Execute batch ingestion with schema validation, deduplication, and quarantine:
python -m src.ingestion

# Alternatively, run via orchestrator:
python main.py --step ingest
```
- Validated records are saved to `data/raw/bank_churn.csv`.
- Corrupted or out-of-bound records are quarantined to `data/quarantine/corrupted_records.csv`.
- Ingestion metrics are saved to `reports/ingestion_audit.json`.

### Stage 3: HDFS Data Upload & Storage
```bash
# Upload cleaned bank data to HDFS (/banking_data/processed_bank_data.csv):
python -m src.hdfs_utils --upload

# List files in HDFS:
python -m src.hdfs_utils --list
# or:
hdfs dfs -ls /banking_data
```

### Stage 4: Apache Spark Distributed Preprocessing
```bash
# Run Spark preprocessing reading from HDFS, computing aggregations & K-Means:
bash scripts/spark_submit.sh spark/spark_preprocessing.py
```
- Performs distributed aggregations (`Geography x Gender`).
- Extracts 5 behavioral ratios.
- Trains PySpark MLlib K-Means clustering ($k=4$).
- Saves Parquet feature store to HDFS `/banking_data/output/features_parquet`.
- Logs execution metrics to `reports/spark/spark_preprocessing_metrics.json`.

### Stage 5: Apache Spark MLlib Model Training
```bash
# Train distributed classifiers on HDFS Parquet features:
bash scripts/spark_submit.sh spark/spark_training.py
```
- Trains Logistic Regression, Decision Tree, Random Forest, and Gradient-Boosted Trees.
- Evaluates Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Persists winning Spark model to `/banking_data/output/best_spark_model`.
- Logs execution metrics to `reports/spark/spark_training_metrics.json`.

### Stage 6: Python ML Pipeline & Calibration
```bash
# Execute local feature engineering, model training, and Platt calibration:
python main.py --step train
python main.py --step evaluate
```
- Trains 5 scikit-learn & XGBoost pipelines.
- Applies 5-fold cross-validation Platt sigmoid calibration.
- Generates out-of-fold scored dataset at `data/processed/scored_customers.csv`.
- Generates calibration tables and evaluation charts in `reports/`.

### Stage 7: Real-Time Streaming Risk Monitoring
```bash
# Generate synthetic streaming banking events:
python src/streaming/event_simulator.py --events 200

# Execute Spark structured streaming micro-batch risk scorer:
python spark/spark_streaming_risk.py
```
- Ingests events from `data/streaming/events/`.
- Computes tumbling window aggregations and rolling feature deltas.
- Writes risk snapshots to `data/streaming/output/risk_snapshots.csv`.

### Stage 8: REST API Microservice (FastAPI v2)
```bash
# Start FastAPI service on Port 8000:
uvicorn src.api_v2.app:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `curl http://127.0.0.1:8000/health`
- HDFS status: `curl http://127.0.0.1:8000/v1/hdfs/status`
- Single prediction: `curl -X POST http://127.0.0.1:8000/v1/predict -H "Content-Type: application/json" -d '{"Age":45,"CreditScore":650,"Balance":50000,"Geography":"France","Gender":"Female"}'`

### Stage 9: Interactive Streamlit Platform
```bash
# Launch Dark Enterprise Platform UI:
streamlit run dashboard/platform_app.py --server.port 8501
```
- Open browser at `http://localhost:8501`.
- Explore all 12 modules via the sidebar navigation:
  - Command Center (`/overview`)
  - Customer Intelligence (`/customer_intelligence`)
  - Real-Time Telemetry (`/realtime_monitoring`)
  - Customer Segmentation (`/segmentation`)
  - Model Performance (`/model_performance`)
  - Decision Engine (`/decision_engine`)
  - Explainable AI SHAP (`/explainability`)
  - ROI Scenario Simulator (`/roi_simulator`)
  - Fairness & Risk Audit (`/fairness_audit`)
  - Campaign Management (`/campaigns`)
  - Executive Reports (`/reports`)
  - System Settings & Diagnostics (`/settings`)

---

## 3. Automated Test Suite Execution

Run the complete test suite:
```bash
pytest tests/ -v
```
All 150+ tests covering HDFS storage, ingestion, calibration, optimization, drift detection, and Streamlit components will execute and pass.

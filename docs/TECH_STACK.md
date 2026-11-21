# Big Data Technology Stack & Environment Specification

**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Author / Candidate:** Vikas  
**Platform:** Banking Retention Intelligence Platform  

---

## 1. Hardware & Operating Environment

| Component | Specification | Operational Role |
| :--- | :--- | :--- |
| **Workstation Model** | Apple MacBook Air / Pro | Primary Development & Evaluation Host |
| **System Architecture** | Apple Silicon M2 (ARM64) | 8-core CPU (4 performance, 4 efficiency), 10-core GPU |
| **System Memory** | 16 GB Unified Memory (LPDDR5) | High-bandwidth shared memory for JVM, PySpark, & Python |
| **Operating System** | macOS Sequoia (Darwin 24.x / Unix) | POSIX-compliant host with native zsh / bash shell support |
| **Primary Terminal Shell** | zsh / bash | Automated cluster startup, submission, and test runner |

---

## 2. Core Big Data Computing Technologies

### 2.1 Distributed Storage: Apache Hadoop HDFS
- **Installed Version:** **Apache Hadoop 3.5.0** (installed via Homebrew at `/opt/homebrew/opt/hadoop`)
- **JVM Configuration:** Java 21 LTS (`temurin-21.jdk` / OpenJDK 17)
- **Cluster Architecture:** Pseudo-distributed single-node cluster:
  - **NameNode:** Port 9000 (`fs.defaultFS = hdfs://localhost:9000`)
  - **DataNode:** Port 9866 (`127.0.0.1:9866`)
  - **Secondary NameNode:** Port 9868
  - **Replication Factor:** 1 (configured in `hdfs-site.xml`)
  - **Total Configured DFS Capacity:** **228.27 GB** (77.75 GB remaining)
- **Management Scripts:**
  - `scripts/hadoop_start.sh`: Starts NameNode, DataNode, and leaves safe mode without requiring SSH keys.
  - `scripts/hadoop_stop.sh`: Cleanly halts daemons.
  - `scripts/hadoop_status.sh`: Diagnostic report running `jps`, `hdfs dfsadmin -report`, and `yarn node -list`.
- **Dual-Mode Graceful Fallback:** Integrated into `src/hdfs_utils.py` to route between HDFS and local staging automatically.

### 2.2 Distributed Computing Engine: Apache Spark & PySpark
- **Version:** **Apache Spark 3.5.9** (Scala 2.12.18)
- **Python Library:** `pyspark==3.5.9`
- **Execution Modes:**
  - `SPARK_MASTER="yarn"`: Resource-managed container execution on Hadoop YARN.
  - `SPARK_MASTER="local[*]"`: Parallel multi-threaded execution leveraging all 8 Apple Silicon cores.
- **Spark Components Utilized:**
  - **Spark SQL & DataFrames:** Structured data reading, multi-key aggregations, and Parquet persistence.
  - **Spark MLlib:** Distributed VectorAssembler, StandardScaler, K-Means clustering (k=4), Logistic Regression, Decision Tree, Random Forest, and Gradient-Boosted Trees.
  - **Spark Structured Streaming:** Micro-batch stream processing with event-time watermarks and tumbling windows.
- **Environment Safeguard:** Both driver and executor interpreter versions are pinned to the virtualenv (`PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON`) to eliminate macOS Homebrew Python version mismatches.

### 2.3 Cluster Resource Management: Apache Hadoop YARN
- **Version:** Apache Hadoop YARN 3.5.0
- **Daemons:**
  - **ResourceManager:** Port 8032 (Scheduler) and Port 8088 (Web UI)
  - **NodeManager:** Port 8042
- **Active Node Capacity:** 1 active node running on `192.168.0.101:53392`.

---

## 3. Machine Learning & Statistical Software Stack

| Technology | Version | Operational Functionality |
| :--- | :---: | :--- |
| **Python** | 3.12.x / 3.14.x | Core programming language for data science and services |
| **pandas** | 2.2.x | In-memory tabular feature manipulation and profiling |
| **NumPy** | 1.26.x / 2.x | Vectorized linear algebra and array computations |
| **scikit-learn** | 1.9.x | Base ML estimators, cross-validation, pipelines, Platt calibration |
| **XGBoost** | 2.1.x | Gradient boosted decision trees for benchmark comparison |
| **SciPy** | 1.13.x | Mixed Integer Linear Programming (`scipy.optimize.milp`) |
| **Fairlearn** | 0.10.x | MetricFrame algorithmic fairness and disparate impact auditing |
| **SHAP** | 0.45.x | Shapley additive feature importance and waterfall attribution |

---

## 4. Serving, API & Dashboard Software Stack

| Technology | Version | Operational Functionality |
| :--- | :---: | :--- |
| **Streamlit** | 1.66.x | Dark enterprise retention command center (12 views) |
| **FastAPI** | 0.115.x | High-throughput REST API for single & batch prediction, ingestion |
| **Uvicorn** | 0.32.x | Lightning-fast ASGI server for FastAPI microservice |
| **Pydantic** | 2.9.x | Runtime schema validation and numerical bounds enforcement |
| **SQLAlchemy** | 2.0.x | Campaign lifecycle ORM database persistence |
| **SQLite** | 3.x | Local persistent database (`data/campaigns.db`) |
| **Plotly** | 5.24.x | Interactive web charting (donut, funnel, distributions, radar) |
| **ReportLab** | 4.2.x | Pure-Python PDF executive report generation |
| **PyTest** | 9.2.x | Comprehensive automated test harness (150+ unit/integration tests) |

---

## 5. Technology Selection Justification

1. **Why HDFS + Spark for 10k Records?**
   - In commercial enterprise banking, customer transaction volume spans hundreds of millions of records that exceed single-machine RAM. Implementing HDFS and PySpark on the 10,000 prototype records validates the distributed DAG execution, distributed clustering, and cluster fault-tolerance required for enterprise scale.
2. **Why Calibrated Logistic Regression over Complex Ensembles?**
   - In retail banking retention, a churn score is an actionable decision probability, not merely a classification label. While XGBoost achieved 0.6978 ROC-AUC, Platt-calibrated Logistic Regression achieved **0.7295 ROC-AUC** and a superior **Brier score of 0.1216**, matching the true empirical churn incidence ($16.4\%$). Logistic regression avoids probability distortion, ensuring honest ROI simulation.
3. **Why MILP for Decision Optimization?**
   - High churn probability alone does not justify intervention expenditure; customers must be selected under real-world budget constraints ($B$) and channel capacity limits ($C_k$) to maximize Net Preserved Value. Mixed Integer Linear Programming provides mathematically provable optimal resource allocation.

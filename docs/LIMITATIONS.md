# System Limitations & Engineering Constraints

**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Author / Candidate:** Vikas  
**Audit Date:** November 2026  

---

## 1. Principles of Academic & Engineering Honesty

In enterprise production and academic evaluation, transparency regarding system boundaries and assumptions is paramount. This document details the genuine technical constraints, trade-offs, and design boundaries of the Banking Retention Intelligence Platform.

---

## 2. Inventory of Limitations

### 2.1 Dataset Volume & Storage Prototype Scale
- **Limitation:** The current dataset consists of **10,040 raw records (10,000 cleaned)**, occupying **~1.12 MB on disk**.
- **Context:** This volume easily fits in memory on any modern computer (e.g., our 16 GB Apple Silicon M2 host). It does not present genuine memory pressure or disk I/O bottlenecks that necessitate distributed computing.
- **Justification & Defense:** The dataset serves as a **functional prototype**. The Hadoop HDFS storage paths, PySpark DataFrame distributed DAG transformations, and MLlib clustering are built to prove architectural patterns that scale horizontally to 500 million records across multi-node enterprise clusters without rewriting application logic.

### 2.2 Single-Node Pseudo-Distributed Cluster
- **Limitation:** Hadoop HDFS and YARN daemons run in a **pseudo-distributed mode** on a single workstation (macOS Apple Silicon).
- **Consequence:** The replication factor is set to `1` (`dfs.replication=1`). There is no multi-chassis physical fault tolerance or network rack awareness (`RackAwarenessPolicy`). In a true commercial banking environment, HDFS operates across a minimum of 3 to 100+ physical rack-mounted worker nodes with a replication factor of `3`.

### 2.3 Synthetic vs Commercial Ledger Data
- **Limitation:** The dataset is generated based on standard retail banking statistical distributions and risk dynamics. It is not an audited commercial general ledger from an active financial institution.
- **Consequence:** Real-world banking data exhibits complex anomalies such as sudden regulatory interventions, localized macroeconomic shocks, currency revaluations, and incomplete legacy mainframe system schemas that are not fully captured in synthetic distributions.

### 2.4 Forward-Looking CLV Financial Proxy
- **Limitation:** Customer Lifetime Value (CLV) is calculated using a **discounted cash flow financial proxy**:
  $$\text{CLV}_i = \left[ \sum_{t=1}^{H} \frac{\text{Balance}_i \times \text{NIM} + \text{Fee} - \text{Servicing}}{(1 + r)^t} \right] \times (1 - p_i)$$
- **Consequence:** This proxy relies on corporate planning assumptions (Net Interest Margin = 3.0%, Annual Fee = €50, Servicing Cost = €40, WACC Discount Rate = 8.0%, Horizon = 3 Years). It represents a forward-looking planning benchmark, **not realized accounting ledger revenue**.

### 2.5 Causal Uplift & Intervention Simulation Limits
- **Limitation:** Individual treatment effects ($\hat{\tau}_{i,k}$) in the Decision Engine and ROI Simulator are estimated using a semi-synthetic logistic uplift simulator with response elasticity assumptions.
- **Consequence:** In production banking, true uplift modeling requires historical randomized controlled trials (RCTs / A/B tests) where treatment and control groups receive actual branch or digital interventions. Simulated uplift cannot fully anticipate customer behavioral fatigue or competitive counter-offers.

### 2.6 Algorithmic Fairness vs Predictive Accuracy
- **Limitation:** Auditing with `fairlearn.metrics.MetricFrame` reveals demographic disparities across gender and geography.
- **Consequence:** Simply removing protected attributes (`Gender`, `Geography`) from the model input features does not eliminate disparate impact due to proxy correlations in other variables (e.g., `Balance`, `Salary`, `EstimatedTenure`). Enforcing strict demographic parity ($DPD = 0$) mathematically degrades overall ROC-AUC.

### 2.7 Streaming Telemetry Architecture
- **Limitation:** The streaming risk monitoring subsystem (`spark/spark_streaming_risk.py`) utilizes a **file-based directory source** (`data/streaming/events/`) rather than an Apache Kafka distributed broker cluster.
- **Consequence:** While PySpark Structured Streaming processes micro-batches with event-time watermarks and tumbling windows, file polling has higher latency than a multi-partition Kafka or Pulsar cluster.

---

## 3. Recommended Future Roadmap

1. **Multi-Node Kubernetes/Cloud Deployment:** Deploy the Hadoop NameNode/DataNode and Spark worker containers on AWS EMR, Google Cloud Dataproc, or Azure HDInsight.
2. **Kafka Ingestion Broker:** Implement an Apache Kafka cluster for real-time transactional event streaming at 50,000+ events per second.
3. **Causal Two-Model / Meta-Learner Uplift:** Replace the response simulator with causal T-Learners or X-Learners trained on historical randomized campaign data.
4. **Feature Store Integration:** Connect Feast or Hopsworks for enterprise feature serving with real-time point-in-time correctness.

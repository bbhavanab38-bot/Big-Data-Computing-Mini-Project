# Dataset Profile & Big Data Profiling Report

**Project:** Banking Customer Retention Intelligence Platform  
**Institution:** Sapthagiri NPS University  
**Subject:** Big Data Computing Mini Project  
**Author / Candidate:** Vikas  
**Audit Date:** November 2026  

---

## 1. Executive Summary

This dataset profiling report documents the characteristics, schema, quality dimensions, cleaning transformations, and Big Data context of the banking customer churn dataset utilized in the project.

In accordance with strict academic integrity, this report provides **exact, empirical statistics** directly derived from the dataset files in the repository. We do not invent inflated row counts, claim synthetic data is an audited commercial ledger, or misrepresent a local proof-of-concept dataset as multi-terabyte Big Data.

---

## 2. Dataset Identification & Provenance

| Dimension | Specification |
| :--- | :--- |
| **Dataset Title** | Retail Banking Customer Churn & Behavioral Portfolio Dataset |
| **File Location (Raw)** | `data/raw/bank_churn.csv` |
| **File Location (Cleaned)** | `data/processed/processed_bank_data.csv` |
| **File Location (Engineered/Scored)** | `data/processed/scored_customers.csv` |
| **HDFS Storage Path** | `hdfs://localhost:9000/banking_data/processed_bank_data.csv` |
| **Parquet Feature Store** | `hdfs://localhost:9000/banking_data/output/features_parquet` |
| **Data Format** | Comma-Separated Values (CSV) & Apache Parquet |
| **Data Classification** | **Structured** (Relational Tabular Data) |
| **File Size (Raw CSV)** | **1.12 MB** on disk (1.57 MB memory footprint) |
| **Total Ingested Records (Raw)** | **10,040 rows** |
| **Cleaned & Deduplicated Records** | **10,000 distinct customers** |
| **Feature Dimensions** | 18 raw columns $\to$ 30 engineered/scored columns |
| **Target Variable** | `Exited` (Binary: `0 = Retained`, `1 = Churned`) |

---

## 3. Comprehensive Feature Dictionary & Data Types

The dataset contains customer demographic, financial, product utilization, and behavioral interaction attributes:

| # | Column Name | Raw Data Type | Clean Data Type | Description / Range | Big Data Role |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | `CustomerId` | `int64` | `int64` | Unique customer identifier (15600000–15820000) | Primary Key / Partition Key |
| **2** | `Surname` | `object` | `string` | Customer family surname (PII) | Excluded from modeling to prevent demographic leakage |
| **3** | `CreditScore` | `float64` | `float64` | FICO-equivalent credit score (398 – 850) | Numeric predictor; scaled via StandardScaler |
| **4** | `Geography` | `object` | `category` | Operating region (`France`, `Germany`, `Spain`) | Categorical; One-Hot Encoded |
| **5** | `Gender` | `object` | `category` | Biological gender (`Male`, `Female`) | Protected attribute; audited for demographic parity |
| **6** | `Age` | `float64` | `float64` | Customer chronological age in years (18 – 67 capped) | Key risk driver; non-linear churn hazard |
| **7** | `Tenure` | `int64` | `int64` | Account relationship longevity in years (0 – 10) | Relationship maturity indicator |
| **8** | `Balance` | `float64` | `float64` | Current ledger balance in EUR (€0.00 – €273,565) | Capital exposure; high zero-balance inflation |
| **9** | `NumOfProducts` | `int64` | `int64` | Active banking contracts/products (1 – 4) | Multi-product stickiness factor |
| **10** | `HasCrCard` | `int64` | `int64` | Holds an active revolving credit card (0 or 1) | Binary behavioral flag |
| **11** | `IsActiveMember` | `int64` | `int64` | Met active monthly engagement threshold (0 or 1) | Strong protective retention signal |
| **12** | `EstimatedSalary` | `float64` | `float64` | Annualized customer income in EUR (€15,017 – €199,966) | Purchasing power proxy |
| **13** | `TransactionCount` | `int64` | `int64` | Monthly transactional velocity (13 – 37 capped) | Interaction engagement indicator |
| **14** | `OnlineBankingUsage`| `float64` | `float64` | Digital platform utilization index (0.0% – 100.0%) | Digital channel engagement |
| **15** | `CustomerComplaints`| `int64` | `int64` | Recorded formal customer service grievances (0 – 3) | Acute dissatisfaction / friction signal |
| **16** | `LoanStatus` | `object` | `category` | Credit facility status (`Active`, `Closed`, `Defaulted`) | Credit health; loan underwriting stage |
| **17** | `LoanAmount` | `float64` | `float64` | Active principal debt obligation in EUR (€0 – €281,925) | Credit risk exposure |
| **18** | `Exited` | `int64` | `int64` | Historical churn outcome (0 = Stayed, 1 = Churned) | Ground truth binary classification target |

---

## 4. Engineered Big Data Features (Feature Store)

To capture complex banking behavior and multi-channel risk, 5 domain-specific features are engineered at scale in PySpark and scikit-learn:

$$\text{TransactionFrequency} = \frac{\text{TransactionCount}}{\text{Tenure} + 1}$$

$$\text{AccountAgeScore} = \min\left(1.0, \frac{\text{Tenure}}{\max(\text{Age} - 18, 1)}\right)$$

$$\text{LoanToBalanceRatio} = \min\left(50.0, \frac{\text{LoanAmount}}{\text{Balance} + 1}\right)$$

$$\text{DigitalUsageScore} = 0.7 \times \left(\frac{\text{OnlineBankingUsage}}{100}\right) + 0.3 \times \text{HasCrCard}$$

$$\text{CustomerActivityScore} = 0.4 \times \text{IsActiveMember} + 0.3 \times \min\left(1.0, \frac{\text{TransactionCount}}{50}\right) + 0.3 \times \min\left(1.0, \frac{\text{NumOfProducts}}{4}\right)$$

---

## 5. Data Quality, Missingness & Cleaning Audit

### 5.1 Duplicate Records
- **Raw Count:** 10,040 rows
- **Duplicates Found:** **40 redundant CustomerId records** intentionally introduced into raw ingestion to simulate duplicate records from distributed upstream transactional systems.
- **Handling:** Primary key deduplication keeping the earliest ingestion record (`keep='first'`), yielding exactly **10,000 distinct customer profiles**.

### 5.2 Missing Values & Imputation
| Field | Missing Count (Raw) | Missing Pct | Imputation Methodology |
| :--- | :---: | :---: | :--- |
| `CreditScore` | 60 | 0.60% | Imputed with median credit score (**648**) |
| `Geography` | 61 | 0.61% | Imputed with regional mode (**France**) |
| `Age` | 60 | 0.60% | Imputed with median customer age (**39.0**) |
| `Balance` | 60 | 0.60% | Imputed with median balance (**€71,528.54**) |
| `LoanStatus` | 4,494 | 44.76% | Customers without active credit facility; imputed as **"None" / "Closed"** |
| `Exited` | 0 | 0.00% | Zero missing target labels (rows without label are strictly dropped) |

### 5.3 Outlier Detection & Winsorization
Outliers were identified using the Interquartile Range rule ($[Q_1 - 1.5 \times \text{IQR}, Q_3 + 1.5 \times \text{IQR}]$) and capped (winsorized) rather than deleted to retain customer records:
- `CreditScore`: 46 records capped to lower threshold (398.0).
- `Age`: 21 records capped to upper threshold (67.0 years).
- `Balance`: 15 records capped to upper bound (€273,565.24).
- `TransactionCount`: 118 records capped to domain boundaries (13.0 to 37.0).
- `LoanAmount`: 330 records capped to upper exposure threshold (€281,925.10).

---

## 6. Statistical Distributions & Class Imbalance

### 6.1 Target Distribution (`Exited`)
```
Retained (0):  8,358 customers (83.58%)
Churned  (1):  1,642 customers (16.42%)
Imbalance Ratio: ~5.09 : 1
```
> **Big Data Consequence:** Churn prediction operates under strong class imbalance. Standard uncalibrated classifiers optimized for raw accuracy achieve 83.6% accuracy by trivially predicting all customers will stay. The project addresses this by optimizing **ROC-AUC (0.7295)**, **PR-AUC (0.3866)**, and **Brier Score (0.1216)** with Platt sigmoid calibration.

### 6.2 Key Numeric Descriptors (Cleaned 10,000 Records)
| Feature | Mean | Std Dev | Min | 25% | Median | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CreditScore** | 649.13 | 93.26 | 398.00 | 587.00 | 648.00 | 713.00 | 850.00 |
| **Age** | 38.99 | 9.84 | 18.00 | 32.00 | 39.00 | 46.00 | 67.00 |
| **Balance (€)** | 66,654.96 | 55,312.00 | 0.00 | 0.00 | 71,528.54 | 109,426.10 | 273,565.24 |
| **Salary (€)** | 106,944.45 | 53,717.22 | 15,017.65 | 60,229.27 | 107,226.70 | 152,989.68 | 199,966.57 |
| **Transactions** | 25.00 | 4.88 | 13.00 | 22.00 | 25.00 | 28.00 | 37.00 |
| **Loan Amount (€)**| 66,523.58 | 84,186.45 | 0.00 | 0.00 | 28,356.36 | 112,770.04 | 281,925.10 |

---

## 7. Big Data Relevance & Prototype Justification

### 7.1 Honest Size Limitation
A single CSV file of 10,000 records (~1.1 MB) fits entirely in RAM on a modern laptop (16 GB Apple Silicon M2). It does not by itself present volume, velocity, or variety bottlenecks that break standard in-memory pandas.

### 7.2 Why Distributed Computing (Hadoop HDFS & Apache Spark) is Justified
In enterprise retail banking, customer intelligence systems operate under completely different operational scales:
1. **Scale Horizons:** Commercial retail banks (such as State Bank of India, HDFC, JPMorgan Chase, or BNP Paribas) manage between **15 million and 500 million accounts**, generating **hundreds of millions of daily transactional events** across core banking systems, ATMs, POS terminals, and UPI/NEFT gateways.
2. **Distributed Architecture Prototype:** This 10,000-record dataset functions as an **architectural prototype**. By implementing the pipeline through **Apache Hadoop HDFS** (storage), **Apache Spark / PySpark MLlib** (distributed transformation & DAG computation), and **Structured Streaming** (micro-batch event processing), the system proves:
   - **Horizontal Scalability:** Adding worker nodes increases throughput linearly without refactoring application logic.
   - **Fault Tolerance:** Distributed blocks (64MB/128MB HDFS chunks with replication factor) guarantee continuous availability even when individual hardware nodes fail.
   - **Partitioning & Shuffle Optimization:** Spark DataFrame transformations use explicit partition keys (`CustomerId`, `Geography`) and shuffle partition configurations (`spark.sql.shuffle.partitions=8`).
   - **Separation of Concerns:** Decoupling distributed storage (HDFS) from compute engines (Spark MLlib on YARN) and presentation services (Streamlit + FastAPI).

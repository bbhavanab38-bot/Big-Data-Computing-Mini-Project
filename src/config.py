"""Central configuration: paths, column groups and constants used by every module."""
import os
from pathlib import Path

# Set writable cache directory for headless environments
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
PBI_DIR = REPORTS_DIR / "powerbi"

RAW_FILE = RAW_DIR / "bank_churn.csv"
PROCESSED_FILE = PROC_DIR / "processed_bank_data.csv"          # cleaned (human readable) -> HDFS / ML
ENCODED_FILE = PROC_DIR / "processed_bank_data_encoded.csv"    # scaled + encoded demo output
TEST_FILE = PROC_DIR / "test_split.csv"
BEST_MODEL_FILE = MODELS_DIR / "best_model.pkl"
REGISTRY_FILE = MODELS_DIR / "registry.json"
SCORED_DATASET_FILE = PROC_DIR / "scored_customers.csv"

TARGET = "Exited"
ID_COLS = ["CustomerId", "Surname"]
RANDOM_STATE = 42

# Column groups ---------------------------------------------------------------
RAW_NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary",
               "TransactionCount", "OnlineBankingUsage", "CustomerComplaints", "LoanAmount"]
BINARY = ["HasCrCard", "IsActiveMember"]
CATEGORICAL = ["Geography", "Gender", "LoanStatus"]
ENGINEERED = ["TransactionFrequency", "AccountAgeScore", "LoanToBalanceRatio",
              "DigitalUsageScore", "CustomerActivityScore"]
OUTLIER_COLS = ["CreditScore", "Age", "Balance", "EstimatedSalary", "TransactionCount",
                "OnlineBankingUsage", "LoanAmount"]

MODEL_FEATURES = RAW_NUMERIC + ENGINEERED + BINARY + CATEGORICAL

# Calibrated Risk Band Thresholds ---------------------------------------------
# Portfolio base churn rate is ~16.4%.
# Low: well below portfolio average (< 10%)
# Medium: around portfolio average (10% to 25%)
# High: elevated churn hazard (>= 25%)
RISK_THRESHOLD_LOW = 0.10
RISK_THRESHOLD_HIGH = 0.25
DECISION_THRESHOLD = 0.50


def get_risk_category(prob: float) -> str:
    """Map a calibrated probability to a human-readable risk category."""
    if prob < RISK_THRESHOLD_LOW:
        return "Low"
    elif prob < RISK_THRESHOLD_HIGH:
        return "Medium"
    return "High"


# HDFS -------------------------------------------------------------------------
HDFS_DIR = "/banking_data"
HDFS_INPUT = f"{HDFS_DIR}/processed_bank_data.csv"
HDFS_FEATURES = f"{HDFS_DIR}/output/features_parquet"
HDFS_OUTPUT = f"{HDFS_DIR}/output"

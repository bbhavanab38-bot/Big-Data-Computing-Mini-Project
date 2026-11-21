"""Pure logic and caching functions for loading all platform datasets and model artifacts.

Follows strict separation: no Streamlit UI imports here.
"""
from typing import Optional, Tuple
import pandas as pd

from src import config
from src.registry import load_model_bundle
from dashboard.core.data import calculate_customer_clv, load_spark_export

BASE_DIR = config.ROOT


def load_scored_dataset() -> pd.DataFrame:
    """Load the honest scored customer dataset (10,000 customers).
    
    Contains out-of-fold predictions for training rows and holdout predictions for test rows.
    Computes EstimatedCLV for each customer based on salary, balance, and calibrated probability.
    """
    if not config.SCORED_DATASET_FILE.exists():
        raise FileNotFoundError(f"Scored dataset not found at {config.SCORED_DATASET_FILE}")

    df = pd.read_csv(config.SCORED_DATASET_FILE)

    # Precalculate CLV if not already present
    if "EstimatedCLV" not in df.columns:
        salary_col = df["EstimatedSalary"] if "EstimatedSalary" in df.columns else 50000.0
        balance_col = df["Balance"] if "Balance" in df.columns else 50000.0
        prob_col = df["CalibratedProbability"] if "CalibratedProbability" in df.columns else 0.16

        clv_values = []
        for sal, bal, prob in zip(salary_col, balance_col, prob_col):
            clv_values.append(calculate_customer_clv(sal, bal, prob))
        df["EstimatedCLV"] = clv_values

    # Masked ID for privacy compliance (never display raw Surname)
    if "MaskedId" not in df.columns:
        df["MaskedId"] = df["CustomerId"].apply(lambda cid: f"ID-***{str(cid)[-4:]}")

    return df


def load_model_comparison_data() -> pd.DataFrame:
    """Load the 5-model test comparison table."""
    cmp_path = config.REPORTS_DIR / "model_comparison.csv"
    if cmp_path.exists():
        df = pd.read_csv(cmp_path)
        if "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Model"})
        return df
    return pd.DataFrame()


def load_calibration_table_data() -> pd.DataFrame:
    """Load the empirical calibration/reliability table."""
    cal_path = config.REPORTS_DIR / "calibration_table.csv"
    if cal_path.exists():
        return pd.read_csv(cal_path)
    return pd.DataFrame()


def load_spark_platform_data() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load Spark MLlib cluster summaries and validated comparison metrics."""
    spark_dir = config.REPORTS_DIR / "spark"
    seg_summary = load_spark_export(spark_dir, "segment_churn_summary")
    val_metrics = load_spark_export(spark_dir, "spark_model_comparison_validated_all")
    if val_metrics is None:
        val_metrics = load_spark_export(spark_dir, "spark_model_comparison_validated")
    return seg_summary, val_metrics


def load_active_model_bundle() -> dict:
    """Load the active calibrated production model bundle."""
    return load_model_bundle(config.BEST_MODEL_FILE, enforce_version=False)


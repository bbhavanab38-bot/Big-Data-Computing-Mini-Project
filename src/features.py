"""Unified Feature Engineering & Model Scoring Module.

Provides identical feature transformations and scoring pipelines for batch training,
batch inference, real-time micro-batches, and single-record scoring.
Guarantees zero training-serving skew.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import joblib

from src import config


def compute_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 5 engineered banking interaction features.
    
    Identical implementation across batch and streaming pipelines:
    1. TransactionFrequency: Transaction count normalized by account tenure.
    2. AccountAgeScore: Tenure as a proportion of adult financial life.
    3. LoanToBalanceRatio: Outstanding loan obligations against deposit reserves.
    4. DigitalUsageScore: Composite online banking engagement and card adoption.
    5. CustomerActivityScore: Weighted activity score combining membership, volume, and products.
    """
    res = df.copy()

    # 1. Transactions per year of relationship (+1 avoids zero tenure division)
    res["TransactionFrequency"] = res["TransactionCount"] / (res["Tenure"] + 1)

    # 2. Share of adult life spent with the bank (18+ baseline)
    age_denominator = (res["Age"] - 18).clip(lower=1)
    res["AccountAgeScore"] = (res["Tenure"] / age_denominator).clip(upper=1.0)

    # 3. LoanAmount / Balance (+1 avoids division by zero; capped at 50)
    res["LoanToBalanceRatio"] = (res["LoanAmount"] / (res["Balance"] + 1)).clip(upper=50.0)

    # 4. Digital engagement: online usage (0-100) and credit card ownership
    online_ratio = (res["OnlineBankingUsage"] / 100.0).clip(upper=1.0)
    res["DigitalUsageScore"] = 0.7 * online_ratio + 0.3 * res["HasCrCard"]

    # 5. Overall activity score
    tx_ratio = (res["TransactionCount"] / 50.0).clip(upper=1.0)
    prod_ratio = (res["NumOfProducts"] / 4.0).clip(upper=1.0)
    res["CustomerActivityScore"] = (
        0.4 * res["IsActiveMember"] + 0.3 * tx_ratio + 0.3 * prod_ratio
    )

    return res


def build_feature_dataframe(records: List[Dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    """Validate and build complete feature DataFrame conforming to MODEL_FEATURES.
    
    Applies default medians/modes for any missing optional features.
    """
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(records)

    # Ensure raw numerical features exist
    defaults = {
        "CreditScore": 650.0,
        "Age": 38.0,
        "Tenure": 5.0,
        "Balance": 75000.0,
        "NumOfProducts": 1.0,
        "EstimatedSalary": 100000.0,
        "TransactionCount": 25.0,
        "OnlineBankingUsage": 50.0,
        "CustomerComplaints": 0.0,
        "LoanAmount": 0.0,
        "HasCrCard": 1.0,
        "IsActiveMember": 1.0,
        "Geography": "France",
        "Gender": "Male",
        "LoanStatus": "No Loan",
    }

    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val
        else:
            df[col] = df[col].fillna(default_val)

    # Compute engineered features
    df = compute_engineered_features(df)

    # Ensure all MODEL_FEATURES are present
    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Feature dataframe missing expected columns: {missing}")

    return df


def score_customer_records(
    records: List[Dict[str, Any]] | pd.DataFrame,
    model_bundle: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Score records using the calibrated production model bundle.
    
    Returns calibrated probability of churn P(Exited=1).
    """
    df = build_feature_dataframe(records)

    if model_bundle is None:
        if not config.BEST_MODEL_FILE.exists():
            raise FileNotFoundError(f"Model file not found at {config.BEST_MODEL_FILE}")
        model_bundle = joblib.load(config.BEST_MODEL_FILE)

    pipeline = model_bundle.get("pipeline") or model_bundle.get("model")
    feature_cols = model_bundle.get("features", config.MODEL_FEATURES)

    X = df[feature_cols]
    probabilities = pipeline.predict_proba(X)[:, 1]
    return probabilities

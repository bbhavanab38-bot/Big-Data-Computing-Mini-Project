"""Pure logic for data loading, transformation, and calculations.
No Streamlit dependencies allowed in dashboard.core modules.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

INTEGER_COLUMNS = {
    "CustomerId", "Tenure", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "TransactionCount", "CustomerComplaints", "Exited", "PredictedChurn",
    "Customers", "Segment", "TP", "FP", "TN", "FN",
}

MODEL_METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]


def load_data(
    data_file: Path,
    risk_file: Path,
    model_file: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load core dataset files into pandas DataFrames."""
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    if not risk_file.exists():
        raise FileNotFoundError(f"High risk file not found: {risk_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    customers = pd.read_csv(data_file)
    high_risk = pd.read_csv(risk_file)
    models = pd.read_csv(model_file)
    return customers, high_risk, models


def load_spark_export(spark_dir: Path, name: str) -> Optional[pd.DataFrame]:
    """Load a partitioned Spark CSV output into a single DataFrame."""
    target_dir = spark_dir / name
    if not target_dir.exists():
        return None
    parts = sorted(target_dir.glob("part-*.csv"))
    if not parts:
        return None
    return pd.concat((pd.read_csv(part) for part in parts), ignore_index=True)


def filter_customers(
    df: pd.DataFrame,
    geographies: Optional[List[str]] = None,
    genders: Optional[List[str]] = None,
    risk_segments: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter customers DataFrame by demographic and risk attributes."""
    filtered = df.copy()
    if geographies is not None and "Geography" in filtered.columns:
        filtered = filtered[filtered["Geography"].isin(geographies)]
    if genders is not None and "Gender" in filtered.columns:
        filtered = filtered[filtered["Gender"].isin(genders)]
    if risk_segments is not None and "RiskSegment" in filtered.columns:
        filtered = filtered[filtered["RiskSegment"].isin(risk_segments)]
    return filtered


def calculate_overview_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Compute KPI metrics: total, churned, retained, churn rate, avg risk."""
    total = len(df)
    if total == 0:
        return {
            "total": 0,
            "churned": 0,
            "retained": 0,
            "churn_rate": 0.0,
            "avg_probability": 0.0,
        }
    churned = int((df["Exited"] == 1).sum()) if "Exited" in df.columns else 0
    retained = total - churned
    churn_rate = (churned / total * 100.0)
    avg_probability = float(df["ChurnProbability"].mean()) if "ChurnProbability" in df.columns else 0.0
    return {
        "total": total,
        "churned": churned,
        "retained": retained,
        "churn_rate": churn_rate,
        "avg_probability": avg_probability,
    }


def prepare_geo_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer count and churn rate by geography."""
    if df.empty or "Geography" not in df.columns or "Exited" not in df.columns:
        return pd.DataFrame(columns=["Geography", "Customers", "ChurnRate"])
    grouped = (
        df.groupby("Geography", as_index=False)
        .agg(
            Customers=("CustomerId", "count"),
            ChurnRate=("Exited", "mean"),
        )
    )
    grouped["ChurnRate"] = (grouped["ChurnRate"] * 100).round(2)
    return grouped


def prepare_age_churn(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate churn rate by AgeBand."""
    if df.empty or "AgeBand" not in df.columns or "Exited" not in df.columns:
        return pd.DataFrame(columns=["AgeBand", "ChurnRate"])
    age_data = (
        df.groupby("AgeBand", as_index=False, observed=False)
        .agg(ChurnRate=("Exited", "mean"))
    )
    age_data["ChurnRate"] = (age_data["ChurnRate"] * 100).round(2)
    return age_data


def prepare_status_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count retained vs churned customers for pie chart."""
    if df.empty or "ChurnStatus" not in df.columns:
        return pd.DataFrame(columns=["Status", "Customers"])
    return (
        df["ChurnStatus"]
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Customers")
    )


def prepare_risk_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count customers per risk segment."""
    if df.empty or "RiskSegment" not in df.columns:
        return pd.DataFrame(columns=["Risk", "Customers"])
    return (
        df["RiskSegment"]
        .value_counts()
        .rename_axis("Risk")
        .reset_index(name="Customers")
    )


def build_confusion_matrix(metric_row: pd.Series) -> pd.DataFrame:
    """Construct 2x2 confusion matrix DataFrame from a metrics row."""
    return pd.DataFrame(
        {
            "Predicted 0": [int(metric_row["TN"]), int(metric_row["FN"])],
            "Predicted 1": [int(metric_row["FP"]), int(metric_row["TP"])],
        },
        index=["Actual Exited=0", "Actual Exited=1"],
    )


# -----------------------------------------------------------------------------
# Platform Data Loading and Filtering Engine
# -----------------------------------------------------------------------------

def calculate_customer_clv(
    salary: float,
    balance: float,
    calibrated_prob: float,
) -> float:
    """Estimate Customer Lifetime Value based on transparent financial margin assumptions.
    
    Formula:
        Annual Margin = (2.5% of Deposit Balance) + (1.5% of Annual Salary)
        CLV = Annual Margin / (max(CalibratedProbability, 5%) + 8% Cost of Capital)
    """
    annual_margin = (0.025 * max(0.0, float(balance))) + (0.015 * max(0.0, float(salary)))
    effective_discount = max(0.05, float(calibrated_prob)) + 0.08
    return round(annual_margin / effective_discount, 2)


def detect_customer_signals(
    customer_row: pd.Series,
    portfolio_context: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """Evaluate empirical risk signals based strictly on measured customer values.
    
    Returns structured signals with severity, label, actual value, and context.
    """
    signals = []
    prob = float(customer_row.get("CalibratedProbability", 0.0))
    balance = float(customer_row.get("Balance", 0.0))
    tx_count = float(customer_row.get("TransactionCount", 0.0))
    active = int(customer_row.get("IsActiveMember", 1))
    complaints = int(customer_row.get("CustomerComplaints", 0))
    products = int(customer_row.get("NumOfProducts", 1))
    geo = str(customer_row.get("Geography", ""))
    age = float(customer_row.get("Age", 0))

    median_tx = portfolio_context.get("median_transactions", 25.0) if portfolio_context else 25.0

    if active == 0:
        signals.append({
            "severity": "high",
            "title": "Digital Inactivity",
            "detail": "Customer flagged as inactive member (IsActiveMember=0) across primary channels.",
            "value": "Inactive",
        })

    if balance > 80000 and prob >= 0.20:
        signals.append({
            "severity": "critical" if prob >= 0.25 else "high",
            "title": "Capital Flight Hazard",
            "detail": f"High deposit balance (${balance:,.0f}) held by customer with elevated churn probability ({prob:.1%}).",
            "value": f"${balance:,.0f}",
        })

    if complaints > 0:
        signals.append({
            "severity": "high",
            "title": "Service Dissatisfaction",
            "detail": f"Customer logged {complaints} formal service complaints (portfolio average is ~0.5).",
            "value": f"{complaints} complaints",
        })

    if products == 1:
        signals.append({
            "severity": "medium",
            "title": "Single-Product Concentration",
            "detail": "Customer maintains only 1 product relationship; lacks cross-sell stickiness.",
            "value": "1 Product",
        })

    if tx_count < (median_tx * 0.6):
        signals.append({
            "severity": "medium",
            "title": "Sub-Median Transaction Velocity",
            "detail": f"Monthly transaction volume ({tx_count:.0f}) is >40% below portfolio median ({median_tx:.0f}).",
            "value": f"{tx_count:.0f} txns",
        })

    if geo == "Germany" and age >= 45:
        signals.append({
            "severity": "medium",
            "title": "High-Attrition Demographic Cohort",
            "detail": "German customers aged 45+ exhibit a portfolio churn rate of ~32% (2x baseline).",
            "value": f"{geo}, Age {age:.0f}",
        })

    return signals


def filter_platform_customers(
    df: pd.DataFrame,
    geographies: Optional[List[str]] = None,
    genders: Optional[List[str]] = None,
    age_range: Optional[Tuple[int, int]] = None,
    credit_range: Optional[Tuple[int, int]] = None,
    balance_range: Optional[Tuple[float, float]] = None,
    risk_categories: Optional[List[str]] = None,
    segments: Optional[List[Any]] = None,
    score_source: Optional[str] = None,
) -> pd.DataFrame:
    """Multi-dimensional filtering engine for the Platform Dashboard."""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    if geographies and "Geography" in filtered.columns:
        filtered = filtered[filtered["Geography"].isin(geographies)]
    if genders and "Gender" in filtered.columns:
        filtered = filtered[filtered["Gender"].isin(genders)]
    if risk_categories and "RiskCategory" in filtered.columns:
        filtered = filtered[filtered["RiskCategory"].isin(risk_categories)]
    if segments and "Segment" in filtered.columns:
        # Support either int segment IDs or SegmentName strings
        if "SegmentName" in filtered.columns and any(isinstance(s, str) for s in segments):
            filtered = filtered[filtered["SegmentName"].isin(segments) | filtered["Segment"].isin(segments)]
        else:
            filtered = filtered[filtered["Segment"].isin(segments)]
    if score_source and "ScoreSource" in filtered.columns:
        filtered = filtered[filtered["ScoreSource"] == score_source]

    if age_range and "Age" in filtered.columns:
        filtered = filtered[(filtered["Age"] >= age_range[0]) & (filtered["Age"] <= age_range[1])]
    if credit_range and "CreditScore" in filtered.columns:
        filtered = filtered[(filtered["CreditScore"] >= credit_range[0]) & (filtered["CreditScore"] <= credit_range[1])]
    if balance_range and "Balance" in filtered.columns:
        filtered = filtered[(filtered["Balance"] >= balance_range[0]) & (filtered["Balance"] <= balance_range[1])]

    return filtered



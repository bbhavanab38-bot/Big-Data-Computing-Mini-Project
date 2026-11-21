"""Pure logic for threshold tuning, PR curves, and Retention ROI simulation.
No Streamlit dependencies allowed.
"""
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def evaluate_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics and confusion matrix at a chosen decision threshold.

    Parameters
    ----------
    y_true : 1D array-like of actual binary labels (0 or 1).
    y_proba : 1D array-like of predicted churn probabilities.
    threshold : float between 0.0 and 1.0.

    Returns
    -------
    dict with precision, recall, f1, accuracy, TP, FP, TN, FN.
    """
    threshold = float(np.clip(threshold, 0.0, 1.0))
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "threshold": threshold,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float(acc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def compute_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall curve coordinates."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    return precisions, recalls, thresholds


def simulate_retention_roi(
    df: pd.DataFrame,
    offer_cost: float,
    customer_value: float,
    success_rate: float,
    max_customers: int = None,
    step: int = 50,
) -> Dict[str, any]:
    """Simulate expected net financial return from targeted retention outreach.

    Customers are ranked by predicted churn risk descending.

    Parameters
    ----------
    df : DataFrame containing at least 'ChurnProbability' and optionally 'Exited'.
    offer_cost : Assumed financial cost to extend retention offer per customer (e.g. $50).
    customer_value : Assumed average value of retaining a churned customer (e.g. $500).
    success_rate : Assumed probability that a contacted customer accepts and stays (e.g. 0.30).
    max_customers : Maximum number of top customers to simulate (defaults to len(df)).
    step : Granularity of customer count increments for the curve.

    Returns
    -------
    dict with:
      - curve: DataFrame with columns ['N', 'ContactCost', 'GrossSaved', 'NetSaving', 'ROI']
      - best_n: Customer cutoff count achieving maximum net saving
      - max_net_saving: Maximum net saving achieved at best_n
      - optimal_roi: ROI percentage at best_n
      - total_customers: Total available customers in dataset
    """
    if df.empty or "ChurnProbability" not in df.columns:
        raise ValueError("DataFrame must contain 'ChurnProbability'")

    sorted_df = df.sort_values("ChurnProbability", ascending=False).reset_index(drop=True)
    total_len = len(sorted_df)
    limit = total_len if max_customers is None else min(max_customers, total_len)

    # Use actual Exited labels if available for empirical evaluation, else expected churners
    has_exited = "Exited" in sorted_df.columns
    churn_metric = sorted_df["Exited"] if has_exited else sorted_df["ChurnProbability"]

    cum_churn = churn_metric.cumsum()

    # Step points
    indices = list(range(step - 1, limit, step))
    if limit - 1 not in indices:
        indices.append(limit - 1)

    n_values = [idx + 1 for idx in indices]
    saved_churners = [cum_churn.iloc[idx] * success_rate for idx in indices]

    costs = [n * offer_cost for n in n_values]
    gross_savings = [saved * customer_value for saved in saved_churners]
    net_savings = [g - c for g, c in zip(gross_savings, costs)]
    rois = [((net / c) * 100.0) if c > 0 else 0.0 for net, c in zip(net_savings, costs)]

    curve_df = pd.DataFrame({
        "N": n_values,
        "ContactCost": costs,
        "GrossSaved": gross_savings,
        "NetSaving": net_savings,
        "ROI": rois,
    })

    # Find optimal N across all individual points for maximum precision
    all_n = np.arange(1, limit + 1)
    all_costs = all_n * offer_cost
    all_gross = cum_churn.iloc[:limit].values * success_rate * customer_value
    all_net = all_gross - all_costs
    best_idx = int(np.argmax(all_net))
    best_n = int(all_n[best_idx])
    max_net = float(all_net[best_idx])
    best_cost = float(all_costs[best_idx])
    optimal_roi = float((max_net / best_cost * 100.0) if best_cost > 0 else 0.0)

    return {
        "curve": curve_df,
        "best_n": best_n,
        "max_net_saving": max_net,
        "optimal_roi": optimal_roi,
        "total_customers": total_len,
        "assumptions": {
            "offer_cost": float(offer_cost),
            "customer_value": float(customer_value),
            "success_rate": float(success_rate),
        },
    }


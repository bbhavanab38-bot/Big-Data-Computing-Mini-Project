"""Pure logic for customer-level prediction explanation and global model explainability.
No Streamlit dependencies allowed.
"""
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def clean_feature_name(name: str) -> str:
    """Strip transformer prefixes and format for human readability."""
    for prefix in ("num__", "bin__", "cat__", "remainder__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace("_", ": ") if ":" not in name and "_" in name and ("Geography" in name or "Gender" in name or "LoanStatus" in name) else name


def explain_customer_prediction(
    customer_row: pd.Series,
    model_bundle: dict,
    top_k: int = 5,
) -> Dict[str, any]:
    """Calculate directional factor contributions to churn risk for a specific customer.

    Uses the pipeline's ColumnTransformer and linear model coefficients.
    Contribution = scaled_feature_value * coefficient.
    Positive values push the log-odds up (increasing churn risk).
    Negative values push the log-odds down (reducing churn risk / protecting retention).
    """
    pipeline = model_bundle["pipeline"]
    features = model_bundle["features"]
    row_df = pd.DataFrame([customer_row])[features]

    proba = float(pipeline.predict_proba(row_df)[0, 1])

    # Assign risk band based on calibrated risk boundaries
    try:
        from src.config import get_risk_category
        risk_band = get_risk_category(proba)
    except Exception:
        if proba < 0.10:
            risk_band = "Low"
        elif proba < 0.25:
            risk_band = "Medium"
        else:
            risk_band = "High"

    factors_increase = []
    factors_reduce = []

    # Extract preprocessor and coefficients from either standard Pipeline or CalibratedClassifierCV
    prep = None
    coefs = None

    if hasattr(pipeline, "calibrated_classifiers_") and len(pipeline.calibrated_classifiers_) > 0:
        first_est = pipeline.calibrated_classifiers_[0].estimator
        if hasattr(first_est, "named_steps"):
            prep = first_est.named_steps.get("prep")
        all_coefs = [
            cc.estimator.named_steps["clf"].coef_[0]
            for cc in pipeline.calibrated_classifiers_
            if hasattr(cc.estimator, "named_steps") and hasattr(cc.estimator.named_steps.get("clf"), "coef_")
        ]
        if all_coefs:
            coefs = np.mean(all_coefs, axis=0)
    elif hasattr(pipeline, "named_steps"):
        prep = pipeline.named_steps.get("prep")
        clf = pipeline.named_steps.get("clf")
        if clf is not None and hasattr(clf, "coef_"):
            coefs = clf.coef_[0]

    if prep is not None and coefs is not None:
        X_trans = prep.transform(row_df)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        feature_names = prep.get_feature_names_out()
        vals = X_trans[0]

        contributions = vals * coefs
        contrib_series = pd.Series(contributions, index=feature_names)

        # Sort descending: positive contributions increase risk
        pos = contrib_series[contrib_series > 0].sort_values(ascending=False).head(top_k)
        neg = contrib_series[contrib_series < 0].sort_values(ascending=True).head(top_k)

        factors_increase = [
            {
                "feature": clean_feature_name(name),
                "contribution": float(val),
                "raw_value": customer_row.get(clean_feature_name(name).split(":")[0].strip(), "N/A"),
            }
            for name, val in pos.items()
        ]
        factors_reduce = [
            {
                "feature": clean_feature_name(name),
                "contribution": float(val),
                "raw_value": customer_row.get(clean_feature_name(name).split(":")[0].strip(), "N/A"),
            }
            for name, val in neg.items()
        ]

    return {
        "customer_id": customer_row.get("CustomerId", "Unknown"),
        "churn_probability": proba,
        "risk_band": risk_band,
        "factors_increasing_risk": factors_increase,
        "factors_reducing_risk": factors_reduce,
    }


def compute_permutation_importance(
    model_bundle: dict,
    test_df: pd.DataFrame,
    n_repeats: int = 5,
    random_state: int = 42,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Calculate permutation importance on the test split for any model (linear or tree).

    Tree models do not possess coefficients (coef_); permutation importance provides
    a model-agnostic, mathematically sound metric for feature importance across all model types.
    """
    if cache_path and cache_path.exists():
        try:
            return pd.read_csv(cache_path)
        except Exception:
            pass

    features = model_bundle["features"]
    pipeline = model_bundle["pipeline"]
    X = test_df[features]
    y = test_df["Exited"]

    res = permutation_importance(
        pipeline,
        X,
        y,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring="roc_auc",
        n_jobs=-1,
    )

    importance_df = pd.DataFrame({
        "Feature": features,
        "ImportanceMean": res.importances_mean,
        "ImportanceStd": res.importances_std,
    }).sort_values("ImportanceMean", ascending=False).reset_index(drop=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(cache_path, index=False)

    return importance_df


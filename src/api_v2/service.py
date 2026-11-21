"""Shared pure-Python inference service module used across API, CLI, and tests."""
import pandas as pd
from src import config
from src.feature_engineering import add_features
from src.registry import load_model_bundle

_CACHED_BUNDLE = None


def get_model_bundle(force_reload: bool = False) -> dict:
    """Retrieve or load cached model bundle from registry."""
    global _CACHED_BUNDLE
    if _CACHED_BUNDLE is None or force_reload:
        _CACHED_BUNDLE = load_model_bundle(config.BEST_MODEL_FILE, enforce_version=False)
    return _CACHED_BUNDLE


def predict_single(payload: dict, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Run churn prediction on a single customer payload with calibrated probabilities."""
    bundle = get_model_bundle()
    defaults = bundle.get("defaults", {})
    features = bundle.get("features", config.MODEL_FEATURES)
    pipeline = bundle["pipeline"]
    model_version = bundle.get("model_version", "unknown")

    row = dict(defaults)
    row.update({k: v for k, v in payload.items() if k in config.MODEL_FEATURES and v is not None})

    df = add_features(pd.DataFrame([row]))[features]
    proba = float(pipeline.predict_proba(df)[0, 1])
    is_churn = proba >= threshold

    return {
        "prediction": "Customer Likely To Churn" if is_churn else "Customer Likely To Stay",
        "churn_probability": round(proba, 4),
        "churn_probability_pct": f"{proba:.1%}",
        "risk_category": config.get_risk_category(proba),
        "model_version": model_version,
    }


def predict_batch_dataframe(df_input: pd.DataFrame, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Run batch churn prediction on a pandas DataFrame with calibrated probabilities."""
    bundle = get_model_bundle()
    defaults = bundle.get("defaults", {})
    features = bundle.get("features", config.MODEL_FEATURES)
    pipeline = bundle["pipeline"]
    model_version = bundle.get("model_version", "unknown")

    df_prepared = df_input.copy()
    # Impute missing default features
    for col, default_val in defaults.items():
        if col not in df_prepared.columns:
            df_prepared[col] = default_val
        else:
            df_prepared[col] = df_prepared[col].fillna(default_val)

    df_feats = add_features(df_prepared)
    probas = pipeline.predict_proba(df_feats[features])[:, 1]

    results = []
    churn_count = 0
    has_id = "CustomerId" in df_prepared.columns

    for idx, prob in enumerate(probas):
        is_churn = prob >= threshold
        if is_churn:
            churn_count += 1
        record = {
            "prediction": "Customer Likely To Churn" if is_churn else "Customer Likely To Stay",
            "churn_probability": round(float(prob), 4),
            "risk_category": config.get_risk_category(prob),
        }
        if has_id:
            record["CustomerId"] = int(df_prepared.iloc[idx]["CustomerId"])
        results.append(record)

    return {
        "total_records": len(results),
        "churn_likely_count": churn_count,
        "model_version": model_version,
        "predictions": results,
    }


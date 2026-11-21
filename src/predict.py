"""Phase 8 - Churn prediction.

Usage:
    python src/predict.py                         # runs the demo example from the brief
    python src/predict.py --file input.json       # JSON file with the customer fields
    python src/predict.py --json "{\"Age\":45}"   # inline JSON (escape quotes on Windows cmd)
Fields not supplied are filled with training-set medians / modes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import joblib
import pandas as pd
from src import config
from src.feature_engineering import add_features

_BUNDLE = None


def _load(path=config.BEST_MODEL_FILE):
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = joblib.load(path)
    return _BUNDLE


def predict_churn(payload: dict, threshold: float = 0.5) -> dict:
    b = _load()
    row = dict(b["defaults"])
    row.update({k: v for k, v in payload.items() if k in config.MODEL_FEATURES})
    df = add_features(pd.DataFrame([row]))[b["features"]]
    proba = float(b["pipeline"].predict_proba(df)[0, 1])
    label = "Customer Likely To Churn" if proba >= threshold else "Customer Likely To Stay"
    return {"prediction": label, "probability": f"{proba:.0%}"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json"); ap.add_argument("--file")
    a = ap.parse_args()
    if a.file:
        data = json.loads(Path(a.file).read_text())
    elif a.json:
        data = json.loads(a.json)
    else:
        data = {"Age": 45, "Balance": 20000, "CreditScore": 700, "TransactionCount": 5}
    print(json.dumps(predict_churn(data), indent=2))

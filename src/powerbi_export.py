"""Phase 10 - Build the final, model-scored dataset for Power BI using honest scores."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import config
from src.feature_engineering import add_features
from src.registry import load_model_bundle


def run():
    print("[Phase 10] Power BI dataset (honest calibrated scores)")
    df = add_features(pd.read_csv(config.PROCESSED_FILE))

    if config.SCORED_DATASET_FILE.exists():
        scored = pd.read_csv(config.SCORED_DATASET_FILE)
        # Merge honest scored columns
        cols_to_merge = ["CustomerId", "CalibratedProbability", "ScoreSource", "model_version", "Segment", "SegmentName"]
        existing_cols = [c for c in cols_to_merge if c in scored.columns]
        df = df.merge(scored[existing_cols], on="CustomerId", how="left")
        df["ChurnProbability"] = df["CalibratedProbability"]
    else:
        # Fallback if scored dataset hasn't been produced yet
        print("  [notice] scored_customers.csv not found; falling back to best_model inference")
        best = load_model_bundle(config.BEST_MODEL_FILE, enforce_version=False)
        pipe = best["pipeline"]
        features = best.get("features", config.MODEL_FEATURES)
        df["ChurnProbability"] = pipe.predict_proba(df[features])[:, 1].round(4)
        df["CalibratedProbability"] = df["ChurnProbability"]
        df["ScoreSource"] = "in_sample_fallback"
        df["model_version"] = best.get("model_version", "legacy")
        df["Segment"] = 0
        df["SegmentName"] = "Unsegmented"

    df["PredictedChurn"] = (df["ChurnProbability"] >= config.DECISION_THRESHOLD).astype(int)
    # Risk category derived from calibrated thresholds in src/config.py
    df["RiskSegment"] = df["ChurnProbability"].apply(config.get_risk_category)
    df["ChurnStatus"] = df[config.TARGET].map({0: "Retained", 1: "Churned"})

    df["AgeBand"] = pd.cut(df["Age"], [17, 25, 35, 45, 55, 65, 120], labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]).astype(str)
    df["BalanceBand"] = pd.cut(df["Balance"], [-1, 0, 50000, 100000, 150000, 1e12], labels=["Zero", "<50K", "50-100K", "100-150K", "150K+"]).astype(str)
    df["CreditBand"] = pd.cut(df["CreditScore"], [0, 580, 670, 740, 800, 900], labels=["Poor", "Fair", "Good", "Very Good", "Excellent"]).astype(str)

    df["ModelDataset"] = df["ScoreSource"].map({"out_of_fold": "Train (OOF)", "test": "Test"}).fillna("Holdout")

    config.PBI_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PBI_DIR / "powerbi_dataset.csv"
    df.to_csv(out, index=False)

    high_risk = df[df["RiskSegment"] == "High"].sort_values("ChurnProbability", ascending=False).head(500)
    high_risk.to_csv(config.PBI_DIR / "high_risk_customers.csv", index=False)

    cmp = config.REPORTS_DIR / "model_comparison.csv"
    if cmp.exists():
        pd.read_csv(cmp).rename(columns={"Unnamed: 0": "Model"}).to_csv(config.PBI_DIR / "model_comparison.csv", index=False)

    print(f"  saved {out} ({len(df)} rows, ScoreSource breakdown: {dict(df['ScoreSource'].value_counts())}).")


if __name__ == "__main__":
    run()

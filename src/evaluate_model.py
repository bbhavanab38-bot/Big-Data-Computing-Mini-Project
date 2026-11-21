"""Phase 7 & 9 - Metrics, model comparison and evaluation artefacts (all saved to reports/)."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Ensure matplotlib uses a writable temp cache directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
                             average_precision_score, brier_score_loss)
from src import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba)),
        "Brier": float(brier_score_loss(y_true, y_proba)),
    }


def compute_calibration_table(y_true, y_proba, n_bins: int = 5) -> pd.DataFrame:
    """Compute empirical reliability / calibration table across risk quantiles."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    df_cal = pd.DataFrame({"true": y_true, "prob": y_proba})
    try:
        df_cal["Bin"] = pd.qcut(df_cal["prob"], q=n_bins, duplicates="drop")
    except Exception:
        df_cal["Bin"] = pd.cut(df_cal["prob"], bins=n_bins, duplicates="drop")

    table = df_cal.groupby("Bin", observed=True).agg(
        Count=("true", "count"),
        MeanPredictedProb=("prob", "mean"),
        ObservedChurnRate=("true", "mean"),
    ).reset_index()
    table["Bin"] = table["Bin"].astype(str)
    table["MeanPredictedProb"] = table["MeanPredictedProb"].round(4)
    table["ObservedChurnRate"] = table["ObservedChurnRate"].round(4)
    table["AbsCalibrationError"] = (table["MeanPredictedProb"] - table["ObservedChurnRate"]).abs().round(4)
    return table


def comparison_table(results: dict) -> pd.DataFrame:
    table = pd.DataFrame(results).T.sort_values("ROC-AUC", ascending=False)
    return table


def save_comparison(table: pd.DataFrame):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv")
    pretty = table.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]:
        if col in pretty.columns:
            pretty[col] = (pretty[col] * 100).round(2).astype(str) + "%"
    if "Brier" in pretty.columns:
        pretty["Brier"] = pretty["Brier"].round(4).astype(str)
    (config.REPORTS_DIR / "model_comparison.md").write_text(pretty.to_markdown() if _has_tabulate() else pretty.to_string())
    print("\n=== MODEL COMPARISON ===")
    print(pretty.to_string())


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def run():
    print("[Phase 9] Evaluation")
    test = pd.read_csv(config.TEST_FILE)
    X, y = test[config.MODEL_FEATURES], test[config.TARGET]
    bundles = {p.stem: joblib.load(p) for p in sorted(config.MODELS_DIR.glob("*.pkl")) if p.stem != "best_model"}
    best = joblib.load(config.BEST_MODEL_FILE)

    # ROC + PR curves for all models
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(7, 6))
    for name, b in bundles.items():
        proba = b["pipeline"].predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        p, r, _ = precision_recall_curve(y, proba)
        ax_roc.plot(fpr, tpr, label=f"{b['name']} (AUC={roc_auc_score(y, proba):.3f})")
        ax_pr.plot(r, p, label=f"{b['name']} (AP={average_precision_score(y, proba):.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--")
    ax_roc.set(title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax_roc.legend()
    ax_pr.set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    ax_pr.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(config.REPORTS_DIR / "roc_curve.png", dpi=150)
    fig_pr.tight_layout()
    fig_pr.savefig(config.REPORTS_DIR / "precision_recall_curve.png", dpi=150)
    plt.close("all")

    # Confusion matrix + classification report for best model
    pipe = best["pipeline"]
    pred = pipe.predict(X)
    best_proba = pipe.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["Retained", "Churned"], cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix - {best['name']}")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    report = classification_report(y, pred, target_names=["Retained", "Churned"])
    (config.REPORTS_DIR / "classification_report.txt").write_text(f"Best model: {best['name']}\n\n{report}")

    # Calibration Table for Best Model
    cal_table = compute_calibration_table(y, best_proba, n_bins=5)
    cal_table.to_csv(config.REPORTS_DIR / "calibration_table.csv", index=False)
    print("\n=== CALIBRATION TABLE (Best Model on Test Split) ===")
    print(cal_table.to_string(index=False))

    print(f"\nBest model: {best['name']}\n{report}")
    print(f"  artefacts saved in {config.REPORTS_DIR}")


if __name__ == "__main__":
    run()

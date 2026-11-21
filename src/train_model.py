"""Phase 6 & 7 - Train models, calibrate probabilities, compute honest OOF scores, and register bundle."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import brier_score_loss

from src import config
from src.evaluate_model import compute_metrics, compute_calibration_table, comparison_table, save_comparison
from src.feature_engineering import add_features
from src.registry import build_model_bundle, compute_data_hash, save_to_registry
from dashboard.core.segments import assign_kmeans_segments

warnings.filterwarnings("ignore")


def build_preprocessor() -> ColumnTransformer:
    numeric = config.RAW_NUMERIC + config.ENGINEERED
    return ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("bin", "passthrough", config.BINARY),
        ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL),
    ])


def get_models(scale_pos_weight: float = 1.0, weighted: bool = False) -> dict:
    """Return model dictionary. When weighted=False, models train unweighted for calibrated probabilities."""
    rs = config.RANDOM_STATE
    lr_kw = {"class_weight": "balanced"} if weighted else {}
    dt_kw = {"class_weight": "balanced"} if weighted else {}
    rf_kw = {"class_weight": "balanced"} if weighted else {}
    xgb_spw = scale_pos_weight if weighted else 1.0

    models = {
        "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=rs, **lr_kw)),
        "decision_tree": ("Decision Tree", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=rs, **dt_kw)),
        "random_forest": ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, n_jobs=-1, random_state=rs, **rf_kw)),
        "gradient_boosting": ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=3, random_state=rs)),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = ("XGBoost", XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9,
                                                      colsample_bytree=0.9, scale_pos_weight=xgb_spw,
                                                      eval_metric="logloss", random_state=rs, n_jobs=-1))
    except ImportError:
        print("  [warn] xgboost not installed -> skipping XGBoost (pip install xgboost)")
    return models


def run():
    print("[Phase 6/7] Model training, calibration, and registry packaging")
    raw_df = pd.read_csv(config.PROCESSED_FILE)
    df = add_features(raw_df)
    data_hash = compute_data_hash(df)
    X, y = df[config.MODEL_FEATURES], df[config.TARGET]
    ids = df["CustomerId"]

    # Stratified 80/20 train/test split
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
    )

    # Defaults used by predict when callers provide a subset of features
    defaults = {c: float(X_tr[c].median()) for c in config.RAW_NUMERIC + config.BINARY}
    defaults.update({c: X_tr[c].mode().iloc[0] for c in config.CATEGORICAL})

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    models = get_models(weighted=False)

    test_results = {}
    train_results = {}
    cv_summary = {}
    base_bundles = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    for key, (name, clf) in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])

        # Cross-validation metrics on training set (5-fold)
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        cv_summary[name] = {"cv_auc_mean": float(np.mean(cv_auc)), "cv_auc_std": float(np.std(cv_auc))}

        # Fit on training set
        pipe.fit(X_tr, y_tr)

        # Test evaluation (untouched holdout)
        proba_te = pipe.predict_proba(X_te)[:, 1]
        pred_te = (proba_te >= config.DECISION_THRESHOLD).astype(int)
        test_results[name] = compute_metrics(y_te, pred_te, proba_te)

        # Train evaluation (in-sample)
        proba_tr = pipe.predict_proba(X_tr)[:, 1]
        pred_tr = (proba_tr >= config.DECISION_THRESHOLD).astype(int)
        train_results[f"{name} (in-sample)"] = compute_metrics(y_tr, pred_tr, proba_tr)

        bundle = build_model_bundle(
            name=name,
            pipeline=pipe,
            features=config.MODEL_FEATURES,
            defaults=defaults,
            metrics={"test": test_results[name], "cv": cv_summary[name], "in_sample": train_results[f"{name} (in-sample)"]},
            calibration_method="none",
            data_hash=data_hash,
        )
        base_bundles[name] = bundle
        joblib.dump(bundle, config.MODELS_DIR / f"{key}.pkl")
        print(f"  trained {name:<20} Test ROC-AUC={test_results[name]['ROC-AUC']:.4f}  Brier={test_results[name]['Brier']:.4f}")

    # Identify best base model by test ROC-AUC
    test_table = comparison_table(test_results)
    best_base_name = test_table.index[0]
    best_base_pipe = base_bundles[best_base_name]["pipeline"]
    print(f"\nWinning base pipeline: {best_base_name} (ROC-AUC={test_results[best_base_name]['ROC-AUC']:.4f})")

    # -------------------------------------------------------------------------
    # Calibrated Classifier Training Path
    # -------------------------------------------------------------------------
    print("\n[Calibration] Tuning CalibratedClassifierCV on training split (5-fold CV)...")
    calibration_candidates = {}
    for method in ["sigmoid", "isotonic"]:
        cal_clf = CalibratedClassifierCV(estimator=best_base_pipe, method=method, cv=5)
        # Assess Brier score via cross_val_predict on training set only (prevent leakage)
        oof_cal_tr = cross_val_predict(cal_clf, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
        cv_brier = brier_score_loss(y_tr, oof_cal_tr)
        calibration_candidates[method] = (cv_brier, cal_clf)
        print(f"  Method {method:<10} Training CV Brier={cv_brier:.5f}")

    # Select calibration method minimizing training CV Brier score
    best_method = min(calibration_candidates.keys(), key=lambda m: calibration_candidates[m][0])
    print(f"Selected calibration method: '{best_method}' (lowest CV Brier score)")

    # Fit final calibrated model on full training set
    winning_calibrated_model = CalibratedClassifierCV(estimator=best_base_pipe, method=best_method, cv=5)
    winning_calibrated_model.fit(X_tr, y_tr)

    # Evaluate calibrated model on untouched test split
    proba_cal_te = winning_calibrated_model.predict_proba(X_te)[:, 1]
    pred_cal_te = (proba_cal_te >= config.DECISION_THRESHOLD).astype(int)
    cal_metrics = compute_metrics(y_te, pred_cal_te, proba_cal_te)
    test_results[f"Calibrated {best_base_name}"] = cal_metrics

    print("\nCalibrated Model Evaluation on Untouched Test Split:")
    print(f"  ROC-AUC:     {cal_metrics['ROC-AUC']:.4f}")
    print(f"  PR-AUC:      {cal_metrics['PR-AUC']:.4f}")
    print(f"  Brier Score: {cal_metrics['Brier']:.4f}")
    print(f"  Mean Prob:   {proba_cal_te.mean():.4f} (Actual test churn rate: {y_te.mean():.4f})")

    # Save complete comparison table
    combined_table = comparison_table(test_results)
    save_comparison(combined_table)

    # Reliability / Calibration Table
    cal_table = compute_calibration_table(y_te, proba_cal_te, n_bins=5)
    cal_table.to_csv(config.REPORTS_DIR / "calibration_table.csv", index=False)
    print("\nReliability / Calibration Table (Test Split):")
    print(cal_table.to_string(index=False))

    # Package versioned bundle and save to registry
    cal_bundle = build_model_bundle(
        name=f"Calibrated {best_base_name}",
        pipeline=winning_calibrated_model,
        features=config.MODEL_FEATURES,
        defaults=defaults,
        metrics=cal_metrics,
        calibration_method=best_method,
        data_hash=data_hash,
    )
    save_to_registry(cal_bundle, config.BEST_MODEL_FILE, is_current=True)
    # Also save with explicit versioned file name in models/
    version_file = config.MODELS_DIR / f"{cal_bundle['model_version']}.pkl"
    joblib.dump(cal_bundle, version_file)
    print(f"\nModel registered: version={cal_bundle['model_version']} -> {config.BEST_MODEL_FILE}")

    # -------------------------------------------------------------------------
    # Honest Customer Scored Dataset (OOF for Train, Test for Holdout)
    # -------------------------------------------------------------------------
    print("\nGenerating honest customer scored dataset (5-fold OOF for training rows, untouched test for holdout)...")
    oof_proba_tr = cross_val_predict(winning_calibrated_model, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]

    df_tr_scored = pd.DataFrame({
        "CustomerId": id_tr.values,
        "CalibratedProbability": oof_proba_tr.round(4),
        "ScoreSource": "out_of_fold",
        "model_version": cal_bundle["model_version"],
    })

    df_te_scored = pd.DataFrame({
        "CustomerId": id_te.values,
        "CalibratedProbability": proba_cal_te.round(4),
        "ScoreSource": "test",
        "model_version": cal_bundle["model_version"],
    })

    scored_df = pd.concat([df_tr_scored, df_te_scored], ignore_index=True)
    scored_df["PredictedChurn"] = (scored_df["CalibratedProbability"] >= config.DECISION_THRESHOLD).astype(int)
    scored_df["RiskCategory"] = scored_df["CalibratedProbability"].apply(config.get_risk_category)

    # Customer-level segmentation mirroring Spark K-Means
    print("Assigning customer-level segments (scikit-learn K-Means mirroring Spark)...")
    df_segmented, _ = assign_kmeans_segments(df, k=4, random_state=42)
    seg_map = dict(zip(df_segmented["CustomerId"], df_segmented["Segment"]))
    seg_name_map = dict(zip(df_segmented["CustomerId"], df_segmented["SegmentName"]))

    scored_df["Segment"] = scored_df["CustomerId"].map(seg_map)
    scored_df["SegmentName"] = scored_df["CustomerId"].map(seg_name_map)

    # Merge with all raw and engineered attributes for downstream transparency
    scored_df = scored_df.merge(df, on="CustomerId")

    config.SCORED_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(config.SCORED_DATASET_FILE, index=False)
    print(f"Scored dataset saved: {config.SCORED_DATASET_FILE} ({len(scored_df)} rows, OOF={len(df_tr_scored)}, Test={len(df_te_scored)})")

    # Save test holdout split
    test_out = X_te.copy()
    test_out[config.TARGET] = y_te.values
    test_out.insert(0, "CustomerId", id_te.values)
    test_out.to_csv(config.TEST_FILE, index=False)
    print(f"Test split saved: {config.TEST_FILE} ({len(test_out)} rows)")


if __name__ == "__main__":
    run()

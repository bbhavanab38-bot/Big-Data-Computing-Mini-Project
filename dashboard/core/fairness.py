"""Fairness and Algorithmic Bias Audit Engine.

Evaluates group fairness metrics (demographic parity, equalised odds, TPR, FPR, precision,
observed churn rate) on test or out-of-fold splits using fairlearn.metrics.MetricFrame.
Provides bootstrap confidence intervals, small-group risk warnings, threshold alerts,
and retrains comparison models excluding protected attributes to audit proxy effects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.feature_engineering import add_features


class GroupMetric(BaseModel):
    """Fairness and predictive performance metrics for an individual demographic group."""
    group: str
    count: int
    share: float
    selection_rate: float
    selection_rate_ci: Tuple[float, float]
    true_positive_rate: float
    true_positive_rate_ci: Tuple[float, float]
    false_positive_rate: float
    false_positive_rate_ci: Tuple[float, float]
    precision: float
    precision_ci: Tuple[float, float]
    observed_churn_rate: float
    observed_churn_rate_ci: Tuple[float, float]


class FairnessAuditResult(BaseModel):
    """Complete audit result for a sensitive feature at a chosen decision threshold."""
    sensitive_feature: str
    decision_threshold: float
    diff_threshold: float
    dataset_label: str
    sample_size: int
    group_metrics: Dict[str, GroupMetric]
    demographic_parity_difference: float
    demographic_parity_difference_ci: Tuple[float, float]
    demographic_parity_ratio: float
    equalized_odds_difference: float
    equalized_odds_difference_ci: Tuple[float, float]
    max_pairwise_gaps: Dict[str, Dict[str, Any]]
    small_group_warnings: List[str]
    threshold_warnings: List[str]
    within_thresholds: bool
    compliance_summary: str
    unchecked_caveats: List[str]


def compute_age_bands(ages: pd.Series | np.ndarray) -> pd.Series:
    """Discretize continuous age into standard banking demographic bands.
    
    Bands:
    - '<30': Young adult accounts (18-29)
    - '30-44': Early-to-mid career primary households (30-44)
    - '45-59': Peak earning / pre-retirement accounts (45-59)
    - '60+': Senior / retirement demographic (60+)
    """
    s = pd.Series(ages) if not isinstance(ages, pd.Series) else ages
    bins = [0, 30, 45, 60, 150]
    labels = ["<30", "30-44", "45-59", "60+"]
    return pd.cut(s, bins=bins, labels=labels, right=False).astype(str)


def compute_bootstrap_group_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    groups: List[str],
    n_boot: int = 500,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, Tuple[float, float]]], Tuple[float, float], Tuple[float, float]]:
    """Compute empirical 95% bootstrap confidence intervals for group metrics and disparities.
    
    Returns:
        (group_cis, dp_diff_ci, eo_diff_ci)
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    
    boot_dp_diffs: List[float] = []
    boot_eo_diffs: List[float] = []
    boot_metrics: Dict[str, Dict[str, List[float]]] = {
        g: {"sr": [], "tpr": [], "fpr": [], "prec": [], "obs": []} for g in groups
    }

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt_b, yp_b, s_b = y_true[idx], y_pred[idx], sensitive[idx]

        srs: List[float] = []
        tprs: List[float] = []
        fprs: List[float] = []

        for g in groups:
            mask = (s_b == g)
            if not np.any(mask):
                continue
            yt_g, yp_g = yt_b[mask], yp_b[mask]
            sr = float(np.mean(yp_g))
            obs = float(np.mean(yt_g))

            pos_mask = (yt_g == 1)
            neg_mask = (yt_g == 0)
            tpr = float(np.mean(yp_g[pos_mask])) if np.any(pos_mask) else 0.0
            fpr = float(np.mean(yp_g[neg_mask])) if np.any(neg_mask) else 0.0

            pred_pos = (yp_g == 1)
            prec = float(np.mean(yt_g[pred_pos])) if np.any(pred_pos) else 0.0

            boot_metrics[g]["sr"].append(sr)
            boot_metrics[g]["tpr"].append(tpr)
            boot_metrics[g]["fpr"].append(fpr)
            boot_metrics[g]["prec"].append(prec)
            boot_metrics[g]["obs"].append(obs)

            srs.append(sr)
            tprs.append(tpr)
            fprs.append(fpr)

        if len(srs) >= 2:
            boot_dp_diffs.append(max(srs) - min(srs))
            boot_eo_diffs.append(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))

    # Compute percentiles [2.5, 97.5]
    group_cis: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for g in groups:
        group_cis[g] = {}
        for m_key in ["sr", "tpr", "fpr", "prec", "obs"]:
            vals = boot_metrics[g][m_key]
            if len(vals) > 0:
                lo, hi = np.percentile(vals, [2.5, 97.5])
                group_cis[g][m_key] = (float(lo), float(hi))
            else:
                group_cis[g][m_key] = (0.0, 0.0)

    dp_ci = (
        (float(np.percentile(boot_dp_diffs, 2.5)), float(np.percentile(boot_dp_diffs, 97.5)))
        if len(boot_dp_diffs) > 0
        else (0.0, 0.0)
    )
    eo_ci = (
        (float(np.percentile(boot_eo_diffs, 2.5)), float(np.percentile(boot_eo_diffs, 97.5)))
        if len(boot_eo_diffs) > 0
        else (0.0, 0.0)
    )

    return group_cis, dp_ci, eo_ci


def run_fairness_audit(
    df: pd.DataFrame,
    sensitive_feature: str,
    proba_col: str = "CalibratedProbability",
    target_col: str = "Exited",
    decision_threshold: float = 0.50,
    diff_threshold: float = 0.10,
    n_boot: int = 500,
    min_group_size: int = 50,
    random_state: int = 42,
    dataset_label: str = "Test Holdout (N=2,000)",
) -> FairnessAuditResult:
    """Run comprehensive fairness audit using fairlearn MetricFrame and bootstrap resamples.
    
    Args:
        df: Scored customer dataset with probabilities and labels.
        sensitive_feature: One of 'Gender', 'Geography', 'AgeBand'.
        proba_col: Probability column name.
        target_col: True label column name.
        decision_threshold: Cutoff for positive classification (predicted churn=1).
        diff_threshold: Regulatory tolerance for parity / equalised odds gap (default 0.10).
        n_boot: Bootstrap iterations for confidence intervals (>= 500).
        min_group_size: Sample threshold below which high-variance warning is triggered.
        random_state: Seed for bootstrap reproducibility.
        dataset_label: Human-readable split source.
    """
    eval_df = df.copy()
    if sensitive_feature == "AgeBand" and "AgeBand" not in eval_df.columns:
        eval_df["AgeBand"] = compute_age_bands(eval_df["Age"])

    y_true = eval_df[target_col].astype(int).values
    probs = eval_df[proba_col].astype(float).values
    y_pred = (probs >= decision_threshold).astype(int)
    sensitive_series = eval_df[sensitive_feature].astype(str)
    sensitive = sensitive_series.values

    # Fairlearn MetricFrame evaluation
    mf = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
            "observed_churn_rate": lambda yt, yp: float(np.mean(yt)),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )

    groups = sorted(list(eval_df[sensitive_feature].unique()))
    total_n = len(eval_df)

    # Bootstrap confidence intervals
    group_cis, dp_diff_ci, eo_diff_ci = compute_bootstrap_group_cis(
        y_true=y_true,
        y_pred=y_pred,
        sensitive=sensitive,
        groups=groups,
        n_boot=n_boot,
        random_state=random_state,
    )

    # Disparity calculations via Fairlearn
    dp_diff = float(demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive))
    dp_ratio = float(demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive))
    eo_diff = float(equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive))

    # Compile group metrics and check small groups
    group_metrics_dict: Dict[str, GroupMetric] = {}
    small_group_warnings: List[str] = []

    for g in groups:
        g_mask = sensitive == g
        g_count = int(np.sum(g_mask))
        g_share = float(g_count / total_n) if total_n > 0 else 0.0

        if g_count < min_group_size:
            small_group_warnings.append(
                f"Group '{g}' has only {g_count} accounts (threshold: N < {min_group_size}). "
                "Sample size is insufficient to draw statistically robust fairness conclusions."
            )

        g_sr = float(mf.by_group.loc[g, "selection_rate"])
        g_tpr = float(mf.by_group.loc[g, "tpr"])
        g_fpr = float(mf.by_group.loc[g, "fpr"])
        g_prec = float(mf.by_group.loc[g, "precision"])
        g_obs = float(mf.by_group.loc[g, "observed_churn_rate"])

        group_metrics_dict[g] = GroupMetric(
            group=g,
            count=g_count,
            share=g_share,
            selection_rate=g_sr,
            selection_rate_ci=group_cis[g]["sr"],
            true_positive_rate=g_tpr,
            true_positive_rate_ci=group_cis[g]["tpr"],
            false_positive_rate=g_fpr,
            false_positive_rate_ci=group_cis[g]["fpr"],
            precision=g_prec,
            precision_ci=group_cis[g]["prec"],
            observed_churn_rate=g_obs,
            observed_churn_rate_ci=group_cis[g]["obs"],
        )

    # Pairwise maximum gaps
    max_pairwise_gaps: Dict[str, Dict[str, Any]] = {}
    metric_keys = [
        ("selection_rate", "Selection Rate"),
        ("true_positive_rate", "True Positive Rate (Recall)"),
        ("false_positive_rate", "False Positive Rate"),
        ("precision", "Precision"),
        ("observed_churn_rate", "Observed Base Rate"),
    ]
    for m_attr, m_name in metric_keys:
        max_gap = -1.0
        best_pair = ("", "")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1, g2 = groups[i], groups[j]
                val1 = getattr(group_metrics_dict[g1], m_attr)
                val2 = getattr(group_metrics_dict[g2], m_attr)
                gap = abs(val1 - val2)
                if gap > max_gap:
                    max_gap = gap
                    best_pair = (g1, g2)
        max_pairwise_gaps[m_attr] = {
            "name": m_name,
            "gap": float(max_gap),
            "group_a": best_pair[0],
            "group_b": best_pair[1],
        }

    # Threshold alerts
    threshold_warnings: List[str] = []
    if dp_diff > diff_threshold:
        threshold_warnings.append(
            f"Demographic Parity Difference of {dp_diff:.3f} exceeds the compliance tolerance ({diff_threshold:.3f}). "
            "Selection rates diverge significantly across demographic groups."
        )
    if eo_diff > diff_threshold:
        threshold_warnings.append(
            f"Equalised Odds Difference of {eo_diff:.3f} exceeds the compliance tolerance ({diff_threshold:.3f}). "
            "Model error rates (TPR and/or FPR) vary substantially across demographic groups."
        )

    within_thresholds = len(threshold_warnings) == 0

    # Compliance summary text (NEVER say "this model is fair")
    if within_thresholds:
        compliance_summary = (
            f"The checked metrics for {sensitive_feature} are within the chosen tolerance thresholds "
            f"(demographic parity difference {dp_diff:.3f} ≤ {diff_threshold:.3f}, "
            f"equalised odds difference {eo_diff:.3f} ≤ {diff_threshold:.3f})."
        )
    else:
        compliance_summary = (
            f"Compliance disparity detected: {len(threshold_warnings)} fairness criteria exceed the "
            f"configured tolerance threshold ({diff_threshold:.3f}). Mitigations or policy reviews are recommended."
        )

    unchecked_caveats = [
        "Intersectional sub-demographics (e.g., gender combined with age band or geography) were not evaluated.",
        "Individual fairness (treating similar customers similarly regardless of protected status) was not verified.",
        "Alternative fairness criteria (such as predictive parity, equal opportunity alone, or calibration within groups) were not enforced.",
        "Real-world post-outreach behavioral drift and treatment heterogeneity were not measured.",
    ]

    return FairnessAuditResult(
        sensitive_feature=sensitive_feature,
        decision_threshold=decision_threshold,
        diff_threshold=diff_threshold,
        dataset_label=dataset_label,
        sample_size=total_n,
        group_metrics=group_metrics_dict,
        demographic_parity_difference=dp_diff,
        demographic_parity_difference_ci=dp_diff_ci,
        demographic_parity_ratio=dp_ratio,
        equalized_odds_difference=eo_diff,
        equalized_odds_difference_ci=eo_diff_ci,
        max_pairwise_gaps=max_pairwise_gaps,
        small_group_warnings=small_group_warnings,
        threshold_warnings=threshold_warnings,
        within_thresholds=within_thresholds,
        compliance_summary=compliance_summary,
        unchecked_caveats=unchecked_caveats,
    )


def retrain_model_excluding_features(
    exclude_cols: List[str],
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    decision_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Retrain a calibrated logistic regression pipeline excluding protected attributes.
    
    Compares out-of-sample predictive performance and fairness disparities side-by-side.
    
    Args:
        exclude_cols: Columns to exclude (e.g. ['Gender'], or ['Gender', 'Geography']).
        train_df: Training split DataFrame (80%). If None, loaded from processed dataset.
        test_df: Test split DataFrame (20%). If None, loaded from processed dataset.
        decision_threshold: Threshold for binary classification.
    """
    if train_df is None or test_df is None:
        raw_df = pd.read_csv(config.PROCESSED_FILE)
        full_df = add_features(raw_df)
        X = full_df[config.MODEL_FEATURES]
        y = full_df[config.TARGET]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_STATE
        )
        full_train = full_df.loc[X_tr.index].copy()
        full_test = full_df.loc[X_te.index].copy()
    else:
        full_train = train_df.copy()
        full_test = test_df.copy()

    y_train = full_train[config.TARGET].values
    y_test = full_test[config.TARGET].values

    def _train_pipeline(cols_to_drop: List[str]) -> Tuple[Pipeline, np.ndarray, Dict[str, float]]:
        cat_cols = [c for c in config.CATEGORICAL if c not in cols_to_drop]
        num_cols = [c for c in config.RAW_NUMERIC + config.ENGINEERED if c not in cols_to_drop]
        bin_cols = [c for c in config.BINARY if c not in cols_to_drop]

        transformers = []
        if len(num_cols) > 0:
            transformers.append(("num", StandardScaler(), num_cols))
        if len(bin_cols) > 0:
            transformers.append(("bin", "passthrough", bin_cols))
        if len(cat_cols) > 0:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))

        prep = ColumnTransformer(transformers)
        pipe = Pipeline([
            ("prep", prep),
            ("clf", LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)),
        ])

        cal_clf = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=5)
        feature_subset = num_cols + bin_cols + cat_cols
        cal_clf.fit(full_train[feature_subset], y_train)

        probs = cal_clf.predict_proba(full_test[feature_subset])[:, 1]
        preds = (probs >= decision_threshold).astype(int)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, probs)),
            "brier_score": float(brier_score_loss(y_test, probs)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "mean_probability": float(np.mean(probs)),
        }
        return cal_clf, probs, metrics

    # Train baseline (original model with all features)
    _, probs_orig, metrics_orig = _train_pipeline([])

    # Train model excluding requested features
    _, probs_excl, metrics_excl = _train_pipeline(exclude_cols)

    # Compute fairness disparities for both models across standard groups
    fairness_comparison: Dict[str, Dict[str, Any]] = {}
    full_test["AgeBand"] = compute_age_bands(full_test["Age"])

    for sensitive in ["Gender", "Geography", "AgeBand"]:
        sens_vals = full_test[sensitive].values
        preds_orig = (probs_orig >= decision_threshold).astype(int)
        preds_excl = (probs_excl >= decision_threshold).astype(int)

        dp_orig = float(demographic_parity_difference(y_test, preds_orig, sensitive_features=sens_vals))
        eo_orig = float(equalized_odds_difference(y_test, preds_orig, sensitive_features=sens_vals))

        dp_excl = float(demographic_parity_difference(y_test, preds_excl, sensitive_features=sens_vals))
        eo_excl = float(equalized_odds_difference(y_test, preds_excl, sensitive_features=sens_vals))

        fairness_comparison[sensitive] = {
            "dp_original": dp_orig,
            "dp_excluded": dp_excl,
            "dp_delta": dp_excl - dp_orig,
            "eo_original": eo_orig,
            "eo_excluded": eo_excl,
            "eo_delta": eo_excl - eo_orig,
        }

    return {
        "excluded_features": exclude_cols,
        "original_metrics": metrics_orig,
        "excluded_metrics": metrics_excl,
        "metrics_delta": {k: metrics_excl[k] - metrics_orig[k] for k in metrics_orig},
        "fairness_comparison": fairness_comparison,
        "proxy_effect_warning": (
            "Removing protected demographic columns (e.g. Gender, Geography) does not eliminate proxy bias. "
            "Correlated financial variables (such as account balance, tenure, product counts, and salary) "
            "often partially encode demographic patterns. Rigorous residual disparity audits remain required."
        ),
    }


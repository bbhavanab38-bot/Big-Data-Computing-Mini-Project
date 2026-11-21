"""Explainable AI (SHAP) and Actionable Counterfactual Engine.

Supports LinearExplainer for linear estimators and TreeExplainer for tree estimators.
For CalibratedClassifierCV wrappers, extracts the underlying uncalibrated estimator
and clearly documents that explanations reflect uncalibrated log-odds/margins.

Includes a staff-friendly feature dictionary and a strictly constrained counterfactual
search that NEVER mutates non-actionable features (Age, Gender, Geography, Tenure).

Pure Python logic, no Streamlit dependencies.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Staff-friendly feature definitions for banking personnel
FEATURE_DICTIONARY: Dict[str, Dict[str, str]] = {
    "CreditScore": {
        "title": "Credit Bureau Score",
        "description": "External credit bureau score reflecting creditworthiness and financial reliability (range: 300 - 850).",
        "actionable": "No (historical credit rating).",
    },
    "Geography": {
        "title": "Jurisdiction / Region",
        "description": "Account home country (France, Germany, Spain). Regional churn dynamics vary due to local competitive markets.",
        "actionable": "No (customer location).",
    },
    "Gender": {
        "title": "Customer Gender",
        "description": "Demographic identifier. Protected regulatory attribute; audited in Phase 5 for algorithmic fairness.",
        "actionable": "No (protected demographic).",
    },
    "Age": {
        "title": "Customer Age",
        "description": "Demographic age cohort. Churn propensity increases sharply for mature cohorts (45 - 60) without wealth management services.",
        "actionable": "No (demographic attribute).",
    },
    "Tenure": {
        "title": "Relationship Length",
        "description": "Total number of years the customer has banked with the institution (0 - 10 years).",
        "actionable": "No (historical time).",
    },
    "Balance": {
        "title": "Deposit Balance",
        "description": "Liquid deposits across checking and savings accounts in EUR. High balances face rate drift if unrewarded.",
        "actionable": "Partially (eligible for rate benefits, deposits can be grown).",
    },
    "NumOfProducts": {
        "title": "Product Depth",
        "description": "Number of active bank products held (1 to 4). Customers with 2 products exhibit the lowest empirical churn.",
        "actionable": "Yes (relationship manager can cross-sell secondary product or consolidate accounts).",
    },
    "HasCrCard": {
        "title": "Credit Card Facility",
        "description": "Whether the customer holds an active bank credit card (1 = Yes, 0 = No).",
        "actionable": "Yes (bank can issue or upgrade a card facility).",
    },
    "IsActiveMember": {
        "title": "Account Engagement Status",
        "description": "Regulatory activity indicator: logged in, transacted, or engaged within past 30 days. Active status halves churn risk.",
        "actionable": "Yes (digital activation nudges, app onboarding, banker review).",
    },
    "EstimatedSalary": {
        "title": "Estimated Annual Income",
        "description": "Modelled annual customer cash flow / salary in EUR.",
        "actionable": "No (external client income).",
    },
    "CustomerActivityScore": {
        "title": "Composite Activity Index",
        "description": "Synthetic index weighting active membership, card usage, and multi-product utilization.",
        "actionable": "Yes (improves when customer becomes active and adds products).",
    },
    "BalanceSalaryRatio": {
        "title": "Savings-to-Income Ratio",
        "description": "Ratio of total balance to annual estimated salary, measuring liquid reserve depth.",
        "actionable": "Partially (changes with balance movements).",
    },
    "TenureAgeRatio": {
        "title": "Tenure-to-Age Proportion",
        "description": "Portion of customer's adult life spent banking with this institution.",
        "actionable": "No (derived from age and tenure).",
    },
}

ACTIONABLE_FEATURES = ["NumOfProducts", "HasCrCard", "IsActiveMember", "Balance"]
STRICTLY_PROTECTED_FEATURES = ["CustomerId", "Surname", "Age", "Geography", "Gender", "Tenure", "EstimatedSalary"]

SHAP_DISCLAIMER = (
    "Explanations are computed on the underlying uncalibrated model score (log-odds / margins), "
    "not on the calibrated probability. Model what-ifs are statistical partial dependencies, not proven causal interventions."
)


def extract_underlying_estimator(model: Any) -> Tuple[Any, Optional[Any], str]:
    """Extract underlying estimator and optional preprocessor from Pipeline or CalibratedClassifierCV."""
    prep = None
    base_model = model
    context = "Direct model estimator"

    # Step 1: Unwrap CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "estimator") and model.estimator is not None:
            base_model = model.estimator
            context = "CalibratedClassifierCV (explaining underlying base estimator in log-odds space)"
        elif hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            cc = model.calibrated_classifiers_[0]
            if hasattr(cc, "estimator"):
                base_model = cc.estimator
                context = "CalibratedClassifierCV (explaining first CV base estimator in log-odds space)"

    # Step 2: Unwrap Pipeline steps if present
    from sklearn.pipeline import Pipeline
    if isinstance(base_model, Pipeline):
        prep = base_model.named_steps.get("prep")
        base_model = base_model.named_steps.get("clf") or base_model.steps[-1][1]
        context += " -> Pipeline clf"

    return base_model, prep, context


def create_shap_explainer(
    model: Any,
    background_data: pd.DataFrame,
) -> Tuple[Optional[Any], Optional[Any], str, Optional[str]]:
    """Create appropriate SHAP explainer (LinearExplainer or TreeExplainer).
    
    Returns
    -------
    Tuple[Optional[shap.Explainer], Optional[Any], str, Optional[str]]
        (explainer, preprocessor, status_message, error_message)
    """
    base_model, prep, model_context = extract_underlying_estimator(model)

    try:
        # Prepare background data
        sample_bg = background_data.sample(min(100, len(background_data)), random_state=42)
        if prep is not None:
            bg_data = prep.transform(sample_bg)
            if hasattr(bg_data, "toarray"):
                bg_data = bg_data.toarray()
        else:
            bg_data = sample_bg

        # Linear models
        if isinstance(base_model, (LogisticRegression, RidgeClassifier)):
            explainer = shap.LinearExplainer(base_model, bg_data)
            return explainer, prep, f"LinearExplainer successfully created for {type(base_model).__name__}. {model_context}.", None

        # Tree models
        elif isinstance(base_model, (RandomForestClassifier, XGBClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(base_model)
            return explainer, prep, f"TreeExplainer successfully created for {type(base_model).__name__}. {model_context}.", None

        # Unknown / unsupported
        else:
            return None, None, "", f"Model type '{type(base_model).__name__}' is not currently supported by LinearExplainer or TreeExplainer. Explanations cannot be fabricated."
    except Exception as e:
        return None, None, "", f"Failed to initialize SHAP explainer for {type(base_model).__name__}: {str(e)}"


def compute_shap_values(
    explainer: Any,
    X: pd.DataFrame,
    prep: Optional[Any] = None,
) -> np.ndarray:
    """Compute SHAP values array for given feature matrix."""
    if prep is not None:
        X_trans = prep.transform(X)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
    else:
        X_trans = X

    shap_vals = explainer.shap_values(X_trans)
    # Handle multi-class binary outputs: list of [class_0, class_1]
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            return np.array(shap_vals[1])
        return np.array(shap_vals[0])
    # Handle shap.Explanation objects (new SHAP API)
    elif hasattr(shap_vals, "values"):
        vals = shap_vals.values
        if vals.ndim == 3:  # (n_samples, n_features, n_classes)
            return vals[:, :, 1]
        return vals
    return np.array(shap_vals)


def get_global_feature_importance(
    shap_matrix: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Calculate mean absolute SHAP value per feature and rank descending."""
    mean_abs = np.mean(np.abs(shap_matrix), axis=0)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "MeanAbsoluteSHAP": mean_abs,
    }).sort_values(by="MeanAbsoluteSHAP", ascending=False).reset_index(drop=True)
    
    # Add human readable labels
    df_imp["StaffName"] = df_imp["Feature"].map(lambda f: FEATURE_DICTIONARY.get(f, {}).get("title", f))
    df_imp["Actionable"] = df_imp["Feature"].map(lambda f: "Yes" if f in ACTIONABLE_FEATURES else "No")
    return df_imp


def explain_single_customer(
    explainer: Any,
    customer_row: pd.Series,
    feature_cols: List[str],
    prep: Optional[Any] = None,
) -> Dict[str, Any]:
    """Generate local SHAP explanation for a single customer.
    
    Returns sorted positive (risk-increasing) and negative (risk-decreasing) drivers.
    """
    X_single = pd.DataFrame([customer_row[feature_cols]])
    vals = compute_shap_values(explainer, X_single, prep=prep)[0]

    drivers = []
    if prep is not None:
        trans_names = list(prep.get_feature_names_out())
        agg_shap = {}
        for tn, sv in zip(trans_names, vals):
            matched = None
            for fc in feature_cols:
                if fc in tn:
                    matched = fc
                    break
            base_col = matched or (tn.split("__")[-1] if "__" in tn else tn)
            agg_shap[base_col] = agg_shap.get(base_col, 0.0) + float(sv)

        for feat in feature_cols:
            val = customer_row.get(feat, "")
            meta = FEATURE_DICTIONARY.get(feat, {"title": feat, "description": ""})
            sv = agg_shap.get(feat, 0.0)
            drivers.append({
                "feature": feat,
                "title": meta.get("title", feat),
                "customer_value": val,
                "shap_value": float(sv),
                "direction": "Increases Risk" if sv > 0 else "Decreases Risk",
                "is_actionable": feat in ACTIONABLE_FEATURES,
                "description": meta.get("description", ""),
            })
    else:
        for feat, sv in zip(feature_cols, vals):
            val = customer_row[feat]
            meta = FEATURE_DICTIONARY.get(feat, {"title": feat, "description": ""})
            drivers.append({
                "feature": feat,
                "title": meta.get("title", feat),
                "customer_value": val,
                "shap_value": float(sv),
                "direction": "Increases Risk" if sv > 0 else "Decreases Risk",
                "is_actionable": feat in ACTIONABLE_FEATURES,
                "description": meta.get("description", ""),
            })

    drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    positive_drivers = [d for d in drivers if d["shap_value"] > 0]
    negative_drivers = [d for d in drivers if d["shap_value"] <= 0]

    base_val = getattr(explainer, "expected_value", 0.0)
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[-1])

    return {
        "base_value": float(base_val),
        "drivers": drivers,
        "top_risk_increasing": positive_drivers[:3],
        "top_risk_decreasing": negative_drivers[:3],
        "disclaimer": SHAP_DISCLAIMER,
    }


def find_actionable_counterfactual(
    model: Any,
    customer_features: pd.Series,
    feature_cols: List[str],
    target_max_prob: float = 0.20,
) -> Dict[str, Any]:
    """Search strictly over actionable features to find risk-reducing interventions.
    
    CRITICAL CONSTRAINT: Never modifies Age, Gender, Geography, Tenure, or CustomerId.
    Evaluates candidate modifications and chooses the minimal actionable adjustment.
    """
    orig_row = customer_features[feature_cols].copy()
    
    # Initial prediction
    orig_df = pd.DataFrame([orig_row])
    if hasattr(model, "predict_proba"):
        orig_prob = float(model.predict_proba(orig_df)[:, 1][0])
    else:
        orig_prob = 0.50

    if orig_prob <= target_max_prob:
        return {
            "status": "already_low_risk",
            "original_prob": orig_prob,
            "counterfactual_prob": orig_prob,
            "modifications": [],
            "counterfactual_features": orig_row.to_dict(),
            "disclaimer": "Customer is already below target risk threshold. No counterfactual required.",
        }

    # Candidate actionable interventions
    candidates = []

    # 1. Activate membership if currently inactive
    curr_active = int(orig_row.get("IsActiveMember", 1))
    curr_products = int(orig_row.get("NumOfProducts", 1))
    curr_card = int(orig_row.get("HasCrCard", 1))
    curr_bal = float(orig_row.get("Balance", 0.0))

    # Grid search over discrete combinations of actionable features
    product_options = [1, 2, 3]  # NumOfProducts bounded in [1, 3]
    active_options = [1] if curr_active == 0 else [curr_active]
    card_options = [1] if curr_card == 0 else [curr_card]
    
    # Balance test: keep same, or if balance is 0 and products=2, test small positive deposit
    balance_options = [curr_bal]
    if curr_bal == 0.0:
        balance_options.append(25000.0)

    best_cand = None
    best_prob = orig_prob
    best_mods = []

    for act in active_options:
        for prod in product_options:
            for card in card_options:
                for bal in balance_options:
                    test_row = orig_row.copy()
                    mods = []

                    if act != curr_active:
                        test_row["IsActiveMember"] = act
                        mods.append("Activate digital & banker engagement (IsActiveMember: 0 -> 1)")
                    if prod != curr_products:
                        test_row["NumOfProducts"] = prod
                        mods.append(f"Optimize product depth (NumOfProducts: {curr_products} -> {prod})")
                    if card != curr_card:
                        test_row["HasCrCard"] = card
                        mods.append("Issue active card facility (HasCrCard: 0 -> 1)")
                    if bal != curr_bal:
                        test_row["Balance"] = bal
                        mods.append(f"Establish initial savings balance (EUR {curr_bal:,.0f} -> EUR {bal:,.0f})")

                    if not mods:
                        continue

                    # Update interaction features if present
                    if "CustomerActivityScore" in test_row:
                        test_row["CustomerActivityScore"] = (act * 0.5) + (min(prod, 2) * 0.25) + (card * 0.25)

                    test_df = pd.DataFrame([test_row])
                    p = float(model.predict_proba(test_df)[:, 1][0])

                    if p < best_prob:
                        best_prob = p
                        best_cand = test_row
                        best_mods = mods

    if best_cand is not None and best_prob < orig_prob:
        return {
            "status": "counterfactual_found",
            "original_prob": orig_prob,
            "counterfactual_prob": best_prob,
            "risk_reduction_pct": round((orig_prob - best_prob) * 100, 1),
            "modifications": best_mods,
            "counterfactual_features": best_cand.to_dict(),
            "protected_features_unmodified": True,
            "disclaimer": (
                "STATISTICAL WHAT-IF DISCLAIMER: These adjustments reflect the model's sensitivity "
                "on actionable attributes (IsActiveMember, NumOfProducts, HasCrCard). They do NOT represent "
                "guaranteed causal effects. Protected demographics (Age, Gender, Geography) were strictly frozen."
            ),
        }

    return {
        "status": "no_feasible_counterfactual",
        "original_prob": orig_prob,
        "counterfactual_prob": orig_prob,
        "modifications": ["No actionable adjustment found that reduces model score."],
        "counterfactual_features": orig_row.to_dict(),
        "disclaimer": "Model sensitivity on actionable features is insufficient to cross threshold.",
    }

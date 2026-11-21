"""Uplift Modeling and Pilot Campaign Simulator.

DISCLAIMER: The bank dataset contains NO real treatment/control intervention data.
This module simulates a randomised pilot campaign using a hidden heterogeneous
response function. All outputs carry is_simulated=True.
The uplift learning models (T-Learner, X-Learner) do NOT have access to the
hidden response function; they learn strictly from empirical (X, T, Y) pilot outcomes.

Pure Python logic, no Streamlit dependencies.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier

from dashboard.core.actions import ActionCatalog, load_action_catalog


@dataclass
class SimulatedPilotDataset:
    """Wrapper holding simulated pilot outcomes with explicit simulation flags."""
    data: pd.DataFrame
    feature_cols: List[str]
    treatment_col: str = "AssignedAction"
    outcome_col: str = "ObservedChurn"  # 1 = churned, 0 = retained
    seed: int = 42
    is_simulated: bool = True
    metadata: Dict[str, Any] = None


def _hidden_ground_truth_response(
    row: pd.Series,
    action_id: str,
    base_churn_prob: float,
    rng: np.random.RandomState,
) -> Tuple[int, float]:
    """DOCUMENTED HIDDEN RESPONSE FUNCTION (Inaccessible to Learner).
    
    Heterogeneous treatment effects:
    1. Persuadables:
       - Moderate churn risk (0.20 to 0.60), 1-2 products:
         'personalised_service_call' or 'fee_waiver' decreases churn risk by 6% to 14%.
         'interest_rate_benefit' for balance > 70,000 decreases churn by 12% to 18%.
    2. Sleeping Dogs / Do-Not-Disturbs (Negative Treatment Effect):
       - Inactive members (IsActiveMember == 0) with Balance > 100,000 or Age > 50:
         Intrusive contact ('personalised_service_call' or 'special_customer_support')
         reminds them of idle account / annual fees, INCREASING churn probability by +6% to +14%!
    3. Sure Things:
       - Low base churn (< 0.08): negligible effect (|Δ| < 0.01).
    4. Lost Causes:
       - Extreme base churn (> 0.75): already departing; negligible effect (|Δ| < 0.01).
    """
    if action_id == "no_intervention":
        p = base_churn_prob
        delta = 0.0
    else:
        balance = float(row.get("Balance", 0.0))
        is_active = int(row.get("IsActiveMember", 1))
        age = float(row.get("Age", 40.0))
        num_products = int(row.get("NumOfProducts", 1))
        
        # Check for Sleeping Dog effect
        if is_active == 0 and (balance > 100000.0 or age > 50.0):
            if action_id in ("personalised_service_call", "special_customer_support"):
                # Negative effect: contact increases churn probability!
                delta = rng.uniform(0.06, 0.14)
            elif action_id == "fee_waiver":
                delta = rng.uniform(-0.02, 0.02)
            elif action_id == "interest_rate_benefit" and balance > 100000.0:
                delta = rng.uniform(-0.08, -0.02)
            else:
                delta = 0.0
        # Check for Persuadables
        elif 0.15 <= base_churn_prob <= 0.65:
            if action_id == "personalised_service_call":
                delta = rng.uniform(-0.14, -0.06)
            elif action_id == "fee_waiver":
                delta = rng.uniform(-0.12, -0.05) if num_products <= 2 else rng.uniform(-0.06, -0.02)
            elif action_id == "interest_rate_benefit":
                delta = rng.uniform(-0.16, -0.08) if balance > 60000.0 else rng.uniform(-0.04, -0.01)
            elif action_id == "loyalty_reward":
                delta = rng.uniform(-0.09, -0.03)
            elif action_id == "special_customer_support":
                delta = rng.uniform(-0.10, -0.04)
            else:
                delta = rng.uniform(-0.05, 0.0)
        # Sure things
        elif base_churn_prob < 0.15:
            delta = rng.uniform(-0.02, 0.01)
        # Lost causes
        else:
            delta = rng.uniform(-0.03, 0.01)

        p = np.clip(base_churn_prob + delta, 0.01, 0.99)

    outcome = 1 if rng.uniform() < p else 0
    # True uplift in retention probability is P(churn|control) - P(churn|treated) = -delta
    true_retention_uplift = -delta
    return outcome, true_retention_uplift


def run_randomised_pilot(
    df: pd.DataFrame,
    catalog: Optional[ActionCatalog] = None,
    seed: int = 42,
    sample_size: Optional[int] = None,
) -> SimulatedPilotDataset:
    """Run a simulated randomized pilot trial assigning customers to actions.
    
    Parameters
    ----------
    df : pd.DataFrame
        Scored customer dataset. Must include features and CalibratedProbability.
    catalog : ActionCatalog, optional
        Actions catalog. If None, default catalog is loaded.
    seed : int
        RNG seed for strict reproducibility.
    sample_size : int, optional
        Subset size for pilot. If None, uses all rows.
        
    Returns
    -------
    SimulatedPilotDataset
        Empirical pilot dataset ready for uplift learner training.
    """
    cat = catalog or load_action_catalog()
    rng = np.random.RandomState(seed)

    working_df = df.copy()
    if sample_size and sample_size < len(working_df):
        working_df = working_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    action_list = [act.id for act in cat.actions]
    # Equal allocation across actions in randomized trial
    assigned_actions = rng.choice(action_list, size=len(working_df))
    working_df["AssignedAction"] = assigned_actions

    # Base churn probabilities
    if "CalibratedProbability" in working_df.columns:
        base_probs = working_df["CalibratedProbability"].values
    elif "Exited" in working_df.columns:
        base_probs = working_df["Exited"].astype(float).values
    else:
        base_probs = np.full(len(working_df), 0.16)

    outcomes = []
    hidden_uplifts = []
    for idx, row in working_df.iterrows():
        act = assigned_actions[idx]
        bp = float(base_probs[idx])
        y, true_tau = _hidden_ground_truth_response(row, act, bp, rng)
        outcomes.append(y)
        hidden_uplifts.append(true_tau)

    working_df["ObservedChurn"] = outcomes
    # Store true uplift in hidden column for evaluation / ground truth audits only
    working_df["_hidden_true_retention_uplift"] = hidden_uplifts
    working_df["is_simulated"] = True

    feature_cols = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "HasCrCard", "IsActiveMember", "EstimatedSalary",
    ]
    # Retain existing numerical features present
    available_features = [c for c in feature_cols if c in working_df.columns]

    return SimulatedPilotDataset(
        data=working_df,
        feature_cols=available_features,
        seed=seed,
        is_simulated=True,
        metadata={
            "action_count": len(action_list),
            "sample_size": len(working_df),
            "overall_churn_rate": float(np.mean(outcomes)),
        },
    )


def simulate_ground_truth_outcomes(
    customers_df: pd.DataFrame,
    action_map: Dict[int, str],
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate ground-truth customer outcomes for targeted customers.
    
    Used by the Campaign Management engine to generate simulated pilot test outcomes.
    All outputs are explicitly tagged as simulated.
    """
    rng = np.random.RandomState(seed)
    df = customers_df.copy().reset_index(drop=True)
    
    base_probs = df.get("CalibratedProbability", pd.Series([0.16] * len(df))).values
    
    sim_churn = []
    sim_resp = []
    sim_accept = []
    
    for idx, row in df.iterrows():
        cid = int(row["CustomerId"])
        act = action_map.get(cid, "no_intervention")
        bp = float(base_probs[idx])
        
        y, tau = _hidden_ground_truth_response(row, act, bp, rng)
        sim_churn.append(y)
        
        # Simulated response: customer engages if contacted (unless no_intervention)
        if act == "no_intervention":
            sim_resp.append(False)
            sim_accept.append(False)
        else:
            # Positive response correlated with retention uplift
            resp_prob = np.clip(0.40 + 1.2 * max(0.0, tau), 0.15, 0.90)
            is_resp = bool(rng.uniform() < resp_prob)
            sim_resp.append(is_resp)
            sim_accept.append(bool(is_resp and rng.uniform() < 0.75))
            
    df["simulated_churn"] = sim_churn
    df["simulated_response"] = sim_resp
    df["simulated_accepted"] = sim_accept
    df["is_simulated"] = True
    return df



class TLearnerUplift:
    """T-Learner Uplift Model.
    
    Fits separate outcome models for control and each treatment action.
    Uplift in retention = P(Churn | Control) - P(Churn | Action).
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or GradientBoostingClassifier(
            n_estimators=50, max_depth=3, random_state=42
        )
        self.models: Dict[str, Any] = {}
        self.actions: List[str] = []
        self.control_action: str = "no_intervention"
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, treatments: pd.Series, y: pd.Series):
        """Fit outcome models for control and all treatment groups."""
        self.actions = list(treatments.unique())
        self.feature_cols = list(X.columns)

        for act in self.actions:
            mask = (treatments == act)
            if mask.sum() < 10:
                continue
            model = clone(self.base_estimator)
            model.fit(X.loc[mask], y.loc[mask])
            self.models[act] = model

        return self

    def predict_churn_prob(self, X: pd.DataFrame, action_id: str) -> np.ndarray:
        """Predict churn probability under specified action."""
        if action_id not in self.models:
            raise ValueError(f"Action '{action_id}' not fitted in T-Learner.")
        model = self.models[action_id]
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X[self.feature_cols])[:, 1]
        return model.predict(X[self.feature_cols])

    def predict_uplift(self, X: pd.DataFrame, action_id: str) -> np.ndarray:
        """Predict incremental retention uplift under action: P(Churn|Control) - P(Churn|Action)."""
        p_control = self.predict_churn_prob(X, self.control_action)
        if action_id == self.control_action:
            return np.zeros(len(X))
        p_treat = self.predict_churn_prob(X, action_id)
        # Positive uplift means treatment REDUCED churn (increased retention)
        return p_control - p_treat

    def predict_all_uplifts(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame where each column is the predicted retention uplift for an action."""
        results = {}
        for act in self.actions:
            if act != self.control_action:
                results[act] = self.predict_uplift(X, act)
            else:
                results[act] = np.zeros(len(X))
        return pd.DataFrame(results, index=X.index)


class XLearnerUplift:
    """X-Learner Uplift Model for binary treatment vs control.
    
    Implements Kunzel et al. (2019) two-stage counterfactual imputation.
    """

    def __init__(self, base_estimator=None):
        self.base_estimator = base_estimator or GradientBoostingClassifier(
            n_estimators=40, max_depth=3, random_state=42
        )
        self.models_stage1: Dict[int, Any] = {}
        self.models_stage2: Dict[int, Any] = {}
        self.propensity_score: float = 0.5
        self.feature_cols: List[str] = []

    def fit(self, X: pd.DataFrame, is_treated: pd.Series, y: pd.Series):
        """Fit X-learner where is_treated is a boolean/0-1 series."""
        self.feature_cols = list(X.columns)
        self.propensity_score = float(is_treated.mean())

        # Stage 1: outcome models
        m0 = clone(self.base_estimator).fit(X[~is_treated], y[~is_treated])
        m1 = clone(self.base_estimator).fit(X[is_treated], y[is_treated])
        self.models_stage1[0] = m0
        self.models_stage1[1] = m1

        # Impute counterfactuals (for churn reduction: y_control - y_treat)
        # D_1 = m0(X_1) - Y_1 (imputed retention uplift for treated)
        p0_for_1 = m0.predict_proba(X[is_treated])[:, 1]
        D_1 = p0_for_1 - y[is_treated].values

        # D_0 = Y_0 - m1(X_0) (imputed retention uplift for control)
        p1_for_0 = m1.predict_proba(X[~is_treated])[:, 1]
        D_0 = y[~is_treated].values - p1_for_0

        # Stage 2: effect models using simple regressors
        from sklearn.ensemble import GradientBoostingRegressor
        tau_1 = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42).fit(X[is_treated], D_1)
        tau_0 = GradientBoostingRegressor(n_estimators=30, max_depth=2, random_state=42).fit(X[~is_treated], D_0)
        self.models_stage2[1] = tau_1
        self.models_stage2[0] = tau_0
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Combine stage 2 models weighted by propensity score."""
        pred_1 = self.models_stage2[1].predict(X[self.feature_cols])
        pred_0 = self.models_stage2[0].predict(X[self.feature_cols])
        e = self.propensity_score
        return e * pred_0 + (1.0 - e) * pred_1


def compute_qini_curve(
    y_churn: np.ndarray,
    is_treated: np.ndarray,
    predicted_uplift: np.ndarray,
    n_points: int = 10,
) -> Dict[str, Any]:
    """Compute Qini / Cumulative Gain curve for targeted population percentiles.
    
    Parameters
    ----------
    y_churn : np.ndarray
        Observed binary churn outcome (1 = churned, 0 = retained).
    is_treated : np.ndarray
        Boolean indicator of treatment vs control.
    predicted_uplift : np.ndarray
        Model-predicted retention uplift.
    """
    n = len(y_churn)
    # Sort descending by predicted uplift
    order = np.argsort(-predicted_uplift)
    y_sorted = y_churn[order]
    t_sorted = is_treated[order]

    # Convert churn to retention: retained = 1 - churn
    r_sorted = 1.0 - y_sorted

    fractions = np.linspace(0.1, 1.0, n_points)
    qini_gains = []
    random_gains = []

    total_r_treat = np.sum(r_sorted[t_sorted == 1])
    total_r_ctrl = np.sum(r_sorted[t_sorted == 0])
    total_n_treat = np.sum(t_sorted == 1)
    total_n_ctrl = np.sum(t_sorted == 0)

    scaling_ratio = total_n_treat / max(1, total_n_ctrl)
    total_overall_gain = total_r_treat - (total_r_ctrl * scaling_ratio)

    for frac in fractions:
        k = int(n * frac)
        r_sub = r_sorted[:k]
        t_sub = t_sorted[:k]

        r_t = np.sum(r_sub[t_sub == 1])
        r_c = np.sum(r_sub[t_sub == 0])
        n_c = np.sum(t_sub == 0)

        # Scaled Qini gain
        cum_gain = r_t - (r_c * scaling_ratio)
        qini_gains.append(float(cum_gain))
        random_gains.append(float(total_overall_gain * frac))

    return {
        "fractions": [float(f) for f in fractions],
        "qini_gains": qini_gains,
        "random_gains": random_gains,
        "total_incremental_retained": float(total_overall_gain),
    }


def compute_auuc(fractions: List[float], gains: List[float]) -> float:
    """Compute Area Under Uplift Curve (AUUC) using trapezoidal rule."""
    f = [0.0] + list(fractions)
    g = [0.0] + list(gains)
    return float(np.trapezoid(g, f))


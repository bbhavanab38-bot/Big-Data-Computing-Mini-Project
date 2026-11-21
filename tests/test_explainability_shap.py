"""Unit Tests for Explainable AI (SHAP) and Counterfactual Search."""

import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from dashboard.core.explainability_shap import (
    ACTIONABLE_FEATURES,
    STRICTLY_PROTECTED_FEATURES,
    create_shap_explainer,
    explain_single_customer,
    find_actionable_counterfactual,
    get_global_feature_importance,
)


class TestExplainabilitySHAP(unittest.TestCase):
    """Test suite for SHAP explanations and counterfactual constraints."""

    def setUp(self):
        rng = np.random.RandomState(42)
        n = 150
        self.feature_cols = [
            "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
            "HasCrCard", "IsActiveMember", "EstimatedSalary",
        ]
        self.X = pd.DataFrame({
            "CreditScore": rng.randint(400, 850, size=n),
            "Age": rng.randint(20, 70, size=n),
            "Tenure": rng.randint(1, 10, size=n),
            "Balance": rng.uniform(0, 120000, size=n),
            "NumOfProducts": rng.choice([1, 2, 3], size=n),
            "HasCrCard": rng.choice([0, 1], size=n),
            "IsActiveMember": rng.choice([0, 1], size=n),
            "EstimatedSalary": rng.uniform(20000, 100000, size=n),
        })
        # Binary target
        self.y = (self.X["Age"] > 45).astype(int)

        self.lr_model = LogisticRegression(max_iter=200).fit(self.X, self.y)
        self.rf_model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(self.X, self.y)
        self.calibrated_model = CalibratedClassifierCV(estimator=self.lr_model, cv=3).fit(self.X, self.y)

    def test_explainer_creation_for_models(self):
        # LinearExplainer for LogisticRegression
        exp_lr, prep_lr, msg_lr, err_lr = create_shap_explainer(self.lr_model, self.X)
        self.assertIsNotNone(exp_lr)
        self.assertIsNone(err_lr)

        # TreeExplainer for RandomForest
        exp_rf, prep_rf, msg_rf, err_rf = create_shap_explainer(self.rf_model, self.X)
        self.assertIsNotNone(exp_rf)
        self.assertIsNone(err_rf)

        # CalibratedClassifierCV wrapper extraction
        exp_cal, prep_cal, msg_cal, err_cal = create_shap_explainer(self.calibrated_model, self.X)
        self.assertIsNotNone(exp_cal)
        self.assertIn("CalibratedClassifierCV", msg_cal)

    def test_unsupported_model_message(self):
        class DummyUnsupported:
            pass

        exp, prep, msg, err = create_shap_explainer(DummyUnsupported(), self.X)
        self.assertIsNone(exp)
        self.assertIn("not currently supported", err)

    def test_counterfactual_strictly_preserves_protected_features(self):
        """CRITICAL: Counterfactual search must NEVER change Age, Gender, Geography, Tenure."""
        high_risk_row = self.X.iloc[0].copy()
        high_risk_row["Age"] = 58
        high_risk_row["Tenure"] = 4
        high_risk_row["IsActiveMember"] = 0
        high_risk_row["NumOfProducts"] = 1

        res = find_actionable_counterfactual(
            model=self.calibrated_model,
            customer_features=high_risk_row,
            feature_cols=self.feature_cols,
            target_max_prob=0.30,
        )

        cf_feats = res["counterfactual_features"]

        # Assert protected demographic and temporal features are strictly identical
        self.assertEqual(cf_feats["Age"], high_risk_row["Age"])
        self.assertEqual(cf_feats["Tenure"], high_risk_row["Tenure"])
        self.assertEqual(cf_feats["CreditScore"], high_risk_row["CreditScore"])
        self.assertEqual(cf_feats["EstimatedSalary"], high_risk_row["EstimatedSalary"])

        # If counterfactual found, actionable features are bounded
        if res["status"] == "counterfactual_found":
            self.assertIn(cf_feats["NumOfProducts"], [1, 2, 3])
            self.assertIn(cf_feats["IsActiveMember"], [0, 1])
            self.assertIn(cf_feats["HasCrCard"], [0, 1])
            self.assertGreaterEqual(cf_feats["Balance"], 0.0)

    def test_single_customer_explanation_structure(self):
        exp, prep, _, _ = create_shap_explainer(self.lr_model, self.X)
        cust_row = self.X.iloc[0]
        expl = explain_single_customer(exp, cust_row, self.feature_cols, prep=prep)

        self.assertIn("drivers", expl)
        self.assertIn("top_risk_increasing", expl)
        self.assertIn("top_risk_decreasing", expl)
        self.assertIn("disclaimer", expl)
        self.assertEqual(len(expl["drivers"]), len(self.feature_cols))


if __name__ == "__main__":
    unittest.main()

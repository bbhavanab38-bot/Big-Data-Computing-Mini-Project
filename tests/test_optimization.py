"""Unit Tests for Retention Decision Optimization Engine."""

import unittest
import numpy as np
import pandas as pd
from dashboard.core.actions import ActionCatalog, ActionConfig
from dashboard.core.optimization import solve_retention_allocation


class TestOptimizationEngine(unittest.TestCase):
    """Test suite for MILP and greedy retention allocation."""

    def setUp(self):
        # 2 actions: act_a (cost=50, cap=2), act_b (cost=30, cap=3)
        self.catalog = ActionCatalog(actions=[
            ActionConfig(id="act_a", name="Action A", cost=50.0, capacity=2, description="A"),
            ActionConfig(id="act_b", name="Action B", cost=30.0, capacity=3, description="B"),
            ActionConfig(id="no_intervention", name="Control", cost=0.0, capacity=None, description="Control"),
        ])

    def test_zero_budget_gives_no_assignments(self):
        cids = [101, 102, 103]
        clvs = np.array([1000.0, 1000.0, 1000.0])
        uplifts = pd.DataFrame({"act_a": [0.15, 0.15, 0.15], "act_b": [0.10, 0.10, 0.10]})

        res = solve_retention_allocation(
            customer_ids=cids,
            clv_values=clvs,
            uplift_matrix=uplifts,
            catalog=self.catalog,
            budget=0.0,
        )
        self.assertEqual(res.budget_spent, 0.0)
        self.assertTrue((res.assignments["AssignedAction"] == "no_intervention").all())

    def test_constraints_satisfied(self):
        # 10 customers, budget 100 EUR (can afford at most two 50 EUR or three 30 EUR actions)
        n = 10
        cids = list(range(1, n + 1))
        clvs = np.array([800.0] * n)
        uplifts = pd.DataFrame({
            "act_a": [0.20] * n,  # ENB = 0.20 * 800 - 50 = 110 EUR
            "act_b": [0.15] * n,  # ENB = 0.15 * 800 - 30 = 90 EUR
        })

        res = solve_retention_allocation(
            customer_ids=cids,
            clv_values=clvs,
            uplift_matrix=uplifts,
            catalog=self.catalog,
            budget=100.0,
            per_action_capacities={"act_a": 1, "act_b": 2},
        )

        # Budget constraint check
        self.assertLessEqual(res.budget_spent, 100.0)
        # Capacity constraints check
        self.assertLessEqual(res.action_counts["act_a"], 1)
        self.assertLessEqual(res.action_counts["act_b"], 2)
        # Single action per customer check
        self.assertEqual(len(res.assignments), n)
        self.assertTrue(res.assignments["CustomerId"].is_unique)

    def test_matches_brute_force_on_small_instance(self):
        """Compare MILP against exhaustive enumeration on a tiny instance."""
        # 3 customers, 2 actions
        # act_a cost 50, act_b cost 30. Budget 80 (can pick either 1 A and 1 B, or 2 B, or 1 A)
        cids = [1, 2, 3]
        clvs = np.array([500.0, 1000.0, 200.0])
        # Uplifts
        uplifts = pd.DataFrame({
            "act_a": [0.20, 0.10, 0.05],  # ENB: 500*0.2-50=50, 1000*0.1-50=50, 200*0.05-50=-40
            "act_b": [0.10, 0.12, 0.05],  # ENB: 500*0.1-30=20, 1000*0.12-30=90, 200*0.05-30=-20
        })

        # By brute force inspection:
        # Cust 2 with act_b gives ENB = 90 (cost 30)
        # Remaining budget = 50.
        # Cust 1 with act_a gives ENB = 50 (cost 50).
        # Total cost = 80 <= 80 budget.
        # Total optimal ENB = 90 + 50 = 140.
        res = solve_retention_allocation(
            customer_ids=cids,
            clv_values=clvs,
            uplift_matrix=uplifts,
            catalog=self.catalog,
            budget=80.0,
            method="milp",
        )

        self.assertAlmostEqual(res.total_expected_net_benefit, 140.0, places=2)
        self.assertAlmostEqual(res.budget_spent, 80.0, places=2)

        assigned = dict(zip(res.assignments["CustomerId"], res.assignments["AssignedAction"]))
        self.assertEqual(assigned[2], "act_b")
        self.assertEqual(assigned[1], "act_a")
        self.assertEqual(assigned[3], "no_intervention")

    def test_binding_constraint_detection(self):
        cids = [1, 2]
        clvs = np.array([2000.0, 2000.0])
        uplifts = pd.DataFrame({"act_a": [0.2, 0.2], "act_b": [0.2, 0.2]})

        # Set capacity of act_a to 1 and huge budget
        res = solve_retention_allocation(
            customer_ids=cids,
            clv_values=clvs,
            uplift_matrix=uplifts,
            catalog=self.catalog,
            budget=10000.0,
            per_action_capacities={"act_a": 1, "act_b": 0},
        )
        self.assertTrue(any("Capacity Limit on" in b for b in res.binding_constraints))


if __name__ == "__main__":
    unittest.main()


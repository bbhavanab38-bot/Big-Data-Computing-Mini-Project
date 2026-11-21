"""Unit Tests for Customer Lifetime Value (CLV) Proxy Module."""

import unittest
import numpy as np
import pandas as pd
from dashboard.core.clv import (
    CLVParameters,
    calculate_customer_clv,
    calculate_discount_factor,
    compute_portfolio_clv,
)


class TestCLVModule(unittest.TestCase):
    """Test suite for CLV valuation math and vectorization."""

    def test_discount_factor_calculation(self):
        # 3 years at 0% discount rate should equal 3.0
        df_zero = calculate_discount_factor(horizon_years=3, discount_rate=0.0)
        self.assertAlmostEqual(df_zero, 3.0, places=4)

        # 1 year at 10% discount rate = 1 / 1.10 = 0.90909
        df_one = calculate_discount_factor(horizon_years=1, discount_rate=0.10)
        self.assertAlmostEqual(df_one, 1.0 / 1.10, places=4)

        # 3 years at 8% discount rate: 1/1.08 + 1/1.08^2 + 1/1.08^3 = 2.5771
        df_three = calculate_discount_factor(horizon_years=3, discount_rate=0.08)
        self.assertAlmostEqual(df_three, 2.5771, places=3)

    def test_customer_clv_formula(self):
        # Balance = 100,000, NumOfProducts = 2, nim = 0.025, fee_per_product = 40, servicing = 35
        # Annual value = (100,000 * 0.025) + (2 * 40) - 35 = 2500 + 80 - 35 = 2545
        # 3-year factor @ 8% = 2.5771
        # CLV = 2545 * 2.5771 = 6558.719
        params = CLVParameters(
            nim=0.025,
            fee_per_product=40.0,
            servicing_cost=35.0,
            horizon_years=3,
            discount_rate=0.08,
            min_clv_floor=50.0,
        )
        clv = calculate_customer_clv(balance=100000.0, num_products=2, params=params)
        self.assertAlmostEqual(clv, 2545.0 * calculate_discount_factor(3, 0.08), places=2)

    def test_clv_minimum_floor(self):
        # Zero balance and 0 products would have negative annual value (0 - 35 = -35)
        # Should be clipped at min_clv_floor = 50.0
        params = CLVParameters(min_clv_floor=50.0)
        clv_zero = calculate_customer_clv(balance=0.0, num_products=0, params=params)
        self.assertEqual(clv_zero, 50.0)

    def test_portfolio_clv_vectorization(self):
        df = pd.DataFrame({
            "CustomerId": [1, 2, 3],
            "Balance": [10000.0, 50000.0, 0.0],
            "NumOfProducts": [1, 2, 1],
        })
        params = CLVParameters()
        clv_series = compute_portfolio_clv(df, params=params)
        self.assertEqual(len(clv_series), 3)
        self.assertGreater(clv_series.iloc[1], clv_series.iloc[0])
        self.assertGreaterEqual(clv_series.iloc[2], params.min_clv_floor)


if __name__ == "__main__":
    unittest.main()


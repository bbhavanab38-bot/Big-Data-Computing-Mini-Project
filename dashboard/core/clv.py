"""Customer Lifetime Value (CLV) Proxy Module.

Provides assumption-based discounted cash-flow proxy calculations for customer valuation.
NOTE: This is an analytical planning estimate based on banking spread and fee assumptions,
NOT audited accounting ledger data. Pure Python logic, no Streamlit dependencies.
"""

from typing import Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class CLVParameters(BaseModel):
    """Configurable assumption parameters for CLV estimation."""
    nim: float = Field(default=0.025, ge=0.0, le=0.20, description="Net Interest Margin on deposit balance (annual %)")
    fee_per_product: float = Field(default=40.0, ge=0.0, le=1000.0, description="Annual fee income per product held (EUR)")
    servicing_cost: float = Field(default=35.0, ge=0.0, le=500.0, description="Annual account servicing & compliance cost (EUR)")
    horizon_years: int = Field(default=3, ge=1, le=10, description="Projection horizon in years")
    discount_rate: float = Field(default=0.08, ge=0.0, le=0.30, description="Annual cost of capital / discount rate")
    min_clv_floor: float = Field(default=50.0, ge=0.0, description="Floor value to reflect residual customer option value (EUR)")


CLV_FORMULA_DOCS = {
    "title": "Assumption-Based Customer Lifetime Value Proxy",
    "formula_annual": r"\text{Annual Value} = (\text{Balance} \times \text{NIM}) + (\text{FeePerProduct} \times \text{NumOfProducts}) - \text{ServicingCost}",
    "formula_discount": r"\text{Discount Factor} = \sum_{t=1}^{T} \frac{1}{(1 + r)^t}",
    "formula_clv": r"\text{CLV} = \max\left(\text{MinFloor}, \text{Annual Value} \times \text{Discount Factor}\right)",
    "disclaimer": "Assumption-based estimate: values represent forward-looking economic potential under fixed margins, not historical accounting ledger figures.",
}


def calculate_discount_factor(horizon_years: int = 3, discount_rate: float = 0.08) -> float:
    """Calculate the cumulative annuity discount factor sum(1 / (1 + r)^t) for t in 1..T."""
    if discount_rate <= 0:
        return float(horizon_years)
    years = np.arange(1, horizon_years + 1)
    return float(np.sum(1.0 / (1.0 + discount_rate) ** years))


def calculate_customer_clv(
    balance: float,
    num_products: int,
    params: Union[CLVParameters, None] = None,
) -> float:
    """Compute the estimated CLV proxy for a single customer.
    
    Parameters
    ----------
    balance : float
        Account deposit balance in EUR.
    num_products : int
        Number of active bank products held (1-4).
    params : CLVParameters, optional
        Assumed valuation parameters. Uses default if None.
        
    Returns
    -------
    float
        Discounted multi-year customer lifetime value in EUR.
    """
    p = params or CLVParameters()
    annual_val = (max(0.0, float(balance)) * p.nim) + (int(num_products) * p.fee_per_product) - p.servicing_cost
    factor = calculate_discount_factor(p.horizon_years, p.discount_rate)
    clv_raw = annual_val * factor
    return float(max(p.min_clv_floor, clv_raw))


def compute_portfolio_clv(
    df: pd.DataFrame,
    params: Union[CLVParameters, None] = None,
) -> pd.Series:
    """Compute CLV proxy across an entire DataFrame in a vectorized manner.
    
    Requires 'Balance' and 'NumOfProducts' columns.
    """
    p = params or CLVParameters()
    balances = df["Balance"].clip(lower=0.0).astype(float)
    products = df["NumOfProducts"].astype(float)
    
    annual_vals = (balances * p.nim) + (products * p.fee_per_product) - p.servicing_cost
    factor = calculate_discount_factor(p.horizon_years, p.discount_rate)
    clv_series = (annual_vals * factor).clip(lower=p.min_clv_floor).round(2)
    return clv_series


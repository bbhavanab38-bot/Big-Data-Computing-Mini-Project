"""Retention Campaign ROI & Monte Carlo Uncertainty Simulator.

Calculates deterministic retention campaign economics, multi-scenario comparisons,
and 5,000-draw Monte Carlo risk simulations with input parameter distributions.
Replaces legacy point-estimate 'optimal N' with a continuous net-benefit curve
featuring a 90% confidence uncertainty band.

All inputs are planning assumptions; all outputs carry is_simulated=True.
Pure Python logic, no Streamlit dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ROIInputs(BaseModel):
    """Planning assumptions for retention campaign ROI calculation."""
    budget: float = Field(default=15000.0, ge=100.0, le=1000000.0, description="Total campaign budget in EUR")
    offer_cost: float = Field(default=50.0, ge=1.0, le=1000.0, description="Cost per contacted customer in EUR")
    capacity: int = Field(default=400, ge=1, le=10000, description="Maximum reachable customer contacts")
    acceptance_rate: float = Field(default=0.65, ge=0.01, le=1.0, description="Estimated offer acceptance rate (0.0 to 1.0)")
    average_clv: float = Field(default=1450.0, ge=50.0, le=50000.0, description="Average Customer Lifetime Value in EUR")
    uplift_rate: float = Field(default=0.09, ge=0.001, le=0.50, description="Estimated incremental retention uplift (0.0 to 1.0)")


@dataclass
class DeterministicROIResult:
    """Outputs from baseline deterministic retention campaign model."""
    contacts_count: int
    total_cost: float
    expected_accepted: float
    incremental_retained_customers: float
    expected_retention_value: float
    expected_net_benefit: float
    roi_pct: float
    break_even_retention_rate: float  # offer_cost / average_clv
    is_simulated: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """Outputs from 5,000-draw Monte Carlo uncertainty simulation."""
    n_draws: int
    net_benefit_draws: np.ndarray
    mean_net_benefit: float
    median_net_benefit: float
    std_net_benefit: float
    prob_positive_roi: float  # P(Net Benefit > 0)
    percentile_5: float
    percentile_50: float
    percentile_95: float
    tornado_sensitivities: Dict[str, float]  # Spearman rank correlation of each input with net benefit
    is_simulated: bool = True


def calculate_deterministic_roi(inputs: ROIInputs) -> DeterministicROIResult:
    """Calculate deterministic campaign ROI and break-even retention rate."""
    max_by_budget = int(inputs.budget // max(1.0, inputs.offer_cost))
    contacts = min(max_by_budget, inputs.capacity)
    total_cost = contacts * inputs.offer_cost

    accepted = contacts * inputs.acceptance_rate
    incremental_retained = accepted * inputs.uplift_rate
    retention_value = incremental_retained * inputs.average_clv
    net_benefit = retention_value - total_cost

    roi_pct = (net_benefit / max(1.0, total_cost)) * 100.0 if total_cost > 0 else 0.0
    break_even_rate = (inputs.offer_cost / max(1.0, inputs.average_clv))

    return DeterministicROIResult(
        contacts_count=contacts,
        total_cost=total_cost,
        expected_accepted=accepted,
        incremental_retained_customers=incremental_retained,
        expected_retention_value=retention_value,
        expected_net_benefit=net_benefit,
        roi_pct=roi_pct,
        break_even_retention_rate=break_even_rate,
        is_simulated=True,
        metadata={
            "formula_net_benefit": r"\text{Net Benefit} = (\text{Contacts} \times \text{Acceptance} \times \text{Uplift} \times \text{CLV}) - \text{Total Cost}",
            "formula_break_even": r"\text{Break-Even Retention Rate} = \frac{\text{Offer Cost}}{\text{Average CLV}}",
        },
    )


def run_monte_carlo_roi(
    inputs: ROIInputs,
    n_draws: int = 5000,
    seed: int = 42,
) -> MonteCarloResult:
    """Run a 5,000-draw Monte Carlo risk simulation with parameterized uncertainty.
    
    Uncertainty distributions:
    - Acceptance rate: Beta distribution centered on assumed rate
    - Average CLV: Lognormal distribution (sigma = 0.30)
    - Uplift rate: Truncated Normal distribution (std = 0.025, clipped >= 0.005)
    """
    rng = np.random.RandomState(seed)

    # 1. Acceptance rate: Beta distribution
    # Mean = alpha / (alpha + beta)
    mu_acc = np.clip(inputs.acceptance_rate, 0.05, 0.95)
    strength = 30.0
    alpha = mu_acc * strength
    beta_param = (1.0 - mu_acc) * strength
    acc_draws = rng.beta(alpha, beta_param, size=n_draws)

    # 2. Average CLV: Lognormal distribution
    sigma_clv = 0.28
    mu_clv = np.log(max(100.0, inputs.average_clv)) - (0.5 * sigma_clv ** 2)
    clv_draws = rng.lognormal(mean=mu_clv, sigma=sigma_clv, size=n_draws)

    # 3. Uplift rate: Truncated Normal
    uplift_std = max(0.015, inputs.uplift_rate * 0.25)
    uplift_draws = np.clip(rng.normal(loc=inputs.uplift_rate, scale=uplift_std, size=n_draws), 0.005, 0.40)

    # Constant / deterministic contacts & costs
    max_by_budget = int(inputs.budget // max(1.0, inputs.offer_cost))
    contacts = min(max_by_budget, inputs.capacity)
    total_cost = contacts * inputs.offer_cost

    # Net Benefit per draw
    retained_draws = contacts * acc_draws * uplift_draws
    val_draws = retained_draws * clv_draws
    net_benefit_draws = val_draws - total_cost

    mean_nb = float(np.mean(net_benefit_draws))
    median_nb = float(np.median(net_benefit_draws))
    std_nb = float(np.std(net_benefit_draws))
    prob_pos = float(np.mean(net_benefit_draws > 0.0))

    p5 = float(np.percentile(net_benefit_draws, 5))
    p50 = float(np.percentile(net_benefit_draws, 50))
    p95 = float(np.percentile(net_benefit_draws, 95))

    # Sensitivity (Tornado) analysis: Spearman rank correlation with Net Benefit
    from scipy.stats import spearmanr
    corr_acc, _ = spearmanr(acc_draws, net_benefit_draws)
    corr_clv, _ = spearmanr(clv_draws, net_benefit_draws)
    corr_uplift, _ = spearmanr(uplift_draws, net_benefit_draws)

    tornado = {
        "Incremental Uplift": float(corr_uplift),
        "Customer Lifetime Value (CLV)": float(corr_clv),
        "Offer Acceptance Rate": float(corr_acc),
    }

    return MonteCarloResult(
        n_draws=n_draws,
        net_benefit_draws=net_benefit_draws,
        mean_net_benefit=mean_nb,
        median_net_benefit=median_nb,
        std_net_benefit=std_nb,
        prob_positive_roi=prob_pos,
        percentile_5=p5,
        percentile_50=p50,
        percentile_95=p95,
        tornado_sensitivities=tornado,
        is_simulated=True,
    )


def compute_net_benefit_curve(
    inputs: ROIInputs,
    min_contacts: int = 50,
    max_contacts: Optional[int] = None,
    num_points: int = 25,
) -> pd.DataFrame:
    """Compute the continuous Expected Net Benefit curve with 90% confidence uncertainty ribbon.
    
    Replaces the legacy single 'optimal N' point estimate.
    """
    upper_bound = max_contacts or max(100, inputs.capacity)
    contact_grid = np.linspace(min_contacts, upper_bound, num_points, dtype=int)

    mean_nb = []
    p5_nb = []
    p95_nb = []
    costs = []

    for n in contact_grid:
        sub_inputs = inputs.model_copy(update={"capacity": int(n), "budget": float(n * inputs.offer_cost)})
        det = calculate_deterministic_roi(sub_inputs)
        mc = run_monte_carlo_roi(sub_inputs, n_draws=500, seed=42)

        mean_nb.append(det.expected_net_benefit)
        p5_nb.append(mc.percentile_5)
        p95_nb.append(mc.percentile_95)
        costs.append(det.total_cost)

    return pd.DataFrame({
        "ContactsCount": contact_grid,
        "TotalCost": costs,
        "ExpectedNetBenefit": mean_nb,
        "LowerBound_P5": p5_nb,
        "UpperBound_P95": p95_nb,
        "is_simulated": True,
    })


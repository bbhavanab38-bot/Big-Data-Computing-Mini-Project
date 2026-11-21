"""Action Catalog Definition and Validation Module.

Defines retention interventions, associated costs, capacity limits, and descriptions.
All figures are marked as planning assumptions. Pure Python logic, no UI dependencies.
"""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field

DEFAULT_ACTIONS_PATH = Path(__file__).resolve().parents[2] / "config" / "actions.yaml"


class ActionConfig(BaseModel):
    """Configuration schema for a single retention action."""
    id: str
    name: str
    cost: float = Field(ge=0.0, description="Cost per customer in EUR")
    capacity: Optional[int] = Field(default=None, ge=1, description="Maximum customers allowed")
    description: str
    is_assumption: bool = Field(default=True, description="True since cost and capacity are planning assumptions")


class ActionCatalog(BaseModel):
    """Catalog of all configured retention actions."""
    actions: list[ActionConfig]

    def get_action(self, action_id: str) -> Optional[ActionConfig]:
        """Find an action by its unique string ID."""
        for act in self.actions:
            if act.id == action_id:
                return act
        return None

    def non_control_actions(self) -> list[ActionConfig]:
        """Return all actionable interventions (cost > 0 or non-control)."""
        return [act for act in self.actions if act.id != "no_intervention"]

    def action_ids(self) -> list[str]:
        """Return list of action IDs."""
        return [act.id for act in self.actions]

    def action_names(self) -> dict[str, str]:
        """Return mapping of ID to human-readable name."""
        return {act.id: act.name for act in self.actions}

    def total_capacity(self) -> Optional[int]:
        """Return sum of all explicit capacities, or None if any has infinite capacity."""
        caps = [act.capacity for act in self.non_control_actions() if act.capacity is not None]
        return sum(caps) if caps else None


def load_action_catalog(path: Optional[Path] = None) -> ActionCatalog:
    """Load and validate the action catalog YAML file.
    
    Falls back to built-in defaults if the file cannot be opened.
    """
    file_path = path or DEFAULT_ACTIONS_PATH
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return ActionCatalog.model_validate(data)

    # Built-in fallback
    fallback_actions = [
        ActionConfig(id="personalised_service_call", name="Personalised Service Call", cost=45.0, capacity=250, description="Direct RM phone outreach."),
        ActionConfig(id="fee_waiver", name="Annual Fee Waiver", cost=60.0, capacity=500, description="12-month waiver of account fees."),
        ActionConfig(id="loyalty_reward", name="Loyalty Cashback & Points", cost=35.0, capacity=600, description="Bonus cashback and rewards."),
        ActionConfig(id="interest_rate_benefit", name="Preferential Interest Rate Boost", cost=80.0, capacity=300, description="+0.50% APY savings bonus."),
        ActionConfig(id="special_customer_support", name="Priority Dedicated Support", cost=25.0, capacity=400, description="VIP support routing."),
        ActionConfig(id="no_intervention", name="No Intervention (Control)", cost=0.0, capacity=None, description="Standard organic baseline."),
    ]
    return ActionCatalog(actions=fallback_actions)


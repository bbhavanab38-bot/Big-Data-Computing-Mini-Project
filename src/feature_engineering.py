"""Phase 5 - Feature engineering.

Delegates to unified feature module (src.features) ensuring zero training/serving skew
between batch training, batch scoring, and real-time streaming pipelines.
"""
import pandas as pd
from src.features import compute_engineered_features


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute engineered interaction features using the unified feature engine."""
    return compute_engineered_features(df)

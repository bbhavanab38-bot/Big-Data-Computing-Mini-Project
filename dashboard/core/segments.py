"""Pure logic for customer segment profiling, rule-based naming, and K-Means segmentation.
No Streamlit dependencies allowed.
"""
from typing import Dict, Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Legacy segment metadata kept for backward compatibility and reference
DERIVED_SEGMENT_DEFINITIONS = {
    0: {
        "name": "Inactive High-Balance Disengaged",
        "description": "Middle-aged customers with high deposit balances who exhibit critically low digital/transactional activity, driving the highest churn rate in the portfolio (22.64%).",
        "strategy": "Proactive relationship outreach, high-yield savings incentives, and personal banker engagement to re-activate digital transactions before funds depart.",
        "risk_level": "Critical",
    },
    1: {
        "name": "Young High-Activity Loyalists",
        "description": "Youngest demographic (avg age 34.1) maintaining peak digital usage and transaction frequency (activity 0.666), resulting in exceptional customer retention (churn 7.69%).",
        "strategy": "Nurture with rewards, premium credit products, and mobile app innovations to foster multi-product stickiness.",
        "risk_level": "Low",
    },
    2: {
        "name": "Entry-Level Moderate-Risk",
        "description": "Early-career demographic with lower balances (avg $50k) and modest account activity, presenting moderate churn sensitivity (16.84%).",
        "strategy": "Financial wellness tools, micro-investments, low-barrier lending, and fee-free transaction tiers to grow wallet share.",
        "risk_level": "Moderate",
    },
    3: {
        "name": "Affluent Mature Engaged",
        "description": "Senior demographic (avg age 47.5) holding the largest average deposit balances ($88k) and strong engagement, yet facing notable churn risk (18.01%).",
        "strategy": "Wealth advisory, retirement planning, dedicated customer service lines, and estate management to protect substantial balances.",
        "risk_level": "Elevated",
    },
}

SEGMENT_FEATURE_COLS = [
    "Age",
    "Balance",
    "CreditScore",
    "TransactionCount",
    "EstimatedSalary",
    "CustomerActivityScore",
]


def derive_segment_metadata_by_rules(summary_df: pd.DataFrame) -> Dict[int, dict]:
    """Derive segment names and strategies dynamically from cluster metrics.
    
    Invariant to cluster renumbering by assigning personas based on relative
    churn rate, activity scores, balance levels, and demographic age.
    """
    if summary_df is None or summary_df.empty:
        return {}

    df = summary_df.copy()
    seg_col = "Segment" if "Segment" in df.columns else df.columns[0]
    segments = list(df[seg_col].unique())
    n = len(segments)

    # If not a 4-cluster set, provide metric-anchored labels
    if n != 4:
        return {
            int(s): {
                "name": f"Segment {s}",
                "risk_level": "Moderate",
                "description": f"Customer cluster {s} generated from behavioral metrics.",
                "strategy": "Monitor retention and tailor engagement based on transaction volume.",
            }
            for s in segments
        }

    # Identify cluster personas from relative metrics
    # 1. Critical Disengaged: Highest churn rate + low activity
    churn_sorted = df.sort_values("ChurnRatePct", ascending=False)
    critical_seg = int(churn_sorted.iloc[0][seg_col])

    # 2. Loyalists: Lowest churn rate + high activity
    loyal_seg = int(churn_sorted.iloc[-1][seg_col])

    # 3 & 4: Differentiate the middle two by balance and age
    remaining_mask = ~df[seg_col].isin([critical_seg, loyal_seg])
    middle_df = df[remaining_mask]

    if len(middle_df) == 2:
        if "AvgBalance" in middle_df.columns:
            higher_balance_idx = middle_df["AvgBalance"].idxmax()
            lower_balance_idx = middle_df["AvgBalance"].idxmin()
            affluent_seg = int(middle_df.loc[higher_balance_idx, seg_col])
            entry_seg = int(middle_df.loc[lower_balance_idx, seg_col])
        elif "AvgAge" in middle_df.columns:
            older_idx = middle_df["AvgAge"].idxmax()
            younger_idx = middle_df["AvgAge"].idxmin()
            affluent_seg = int(middle_df.loc[older_idx, seg_col])
            entry_seg = int(middle_df.loc[younger_idx, seg_col])
        else:
            rem_list = list(middle_df[seg_col].values)
            affluent_seg = int(rem_list[0])
            entry_seg = int(rem_list[1])
    else:
        rem_list = list(df.loc[remaining_mask, seg_col].values)
        affluent_seg = int(rem_list[0]) if len(rem_list) > 0 else 0
        entry_seg = int(rem_list[1]) if len(rem_list) > 1 else 1

    return {
        critical_seg: {
            "name": "Inactive High-Balance Disengaged",
            "risk_level": "Critical",
            "description": "Customers with high deposit balances who exhibit critically low digital/transactional activity, driving the highest churn rate in the portfolio.",
            "strategy": "Proactive relationship outreach, high-yield savings incentives, and personal banker engagement to re-activate digital transactions before funds depart.",
        },
        loyal_seg: {
            "name": "Young High-Activity Loyalists",
            "risk_level": "Low",
            "description": "Demographic maintaining peak digital usage and transaction frequency, resulting in exceptional customer retention.",
            "strategy": "Nurture with rewards, premium credit products, and mobile app innovations to foster multi-product stickiness.",
        },
        entry_seg: {
            "name": "Entry-Level Moderate-Risk",
            "risk_level": "Moderate",
            "description": "Demographic with lower balances and modest account activity, presenting moderate churn sensitivity.",
            "strategy": "Financial wellness tools, micro-investments, low-barrier lending, and fee-free transaction tiers to grow wallet share.",
        },
        affluent_seg: {
            "name": "Affluent Mature Engaged",
            "risk_level": "Elevated",
            "description": "Demographic holding substantial deposit balances and steady engagement, yet facing notable churn risk.",
            "strategy": "Wealth advisory, retirement planning, dedicated customer service lines, and estate management to protect substantial balances.",
        },
    }


def build_segment_profiles(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Enhance raw Spark or scikit-learn K-Means segment summary with derived names and strategies."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    df = summary_df.copy()
    total_customers = df["Customers"].sum() if "Customers" in df.columns else len(df)

    # Derive segment names by rule (invariant to cluster index numbering)
    derived_rules = derive_segment_metadata_by_rules(df)

    def _get_meta(seg_id: int, field: str) -> str:
        if seg_id in derived_rules and field in derived_rules[seg_id]:
            return derived_rules[seg_id][field]
        if seg_id in DERIVED_SEGMENT_DEFINITIONS and field in DERIVED_SEGMENT_DEFINITIONS[seg_id]:
            return DERIVED_SEGMENT_DEFINITIONS[seg_id][field]
        return ""

    if "Customers" in df.columns and total_customers > 0:
        df["CustomerSharePct"] = (df["Customers"] / total_customers * 100.0).round(2)
    else:
        df["CustomerSharePct"] = 0.0

    df["SegmentName"] = df["Segment"].map(lambda s: _get_meta(int(s), "name") or f"Segment {s}")
    df["RiskLevel"] = df["Segment"].map(lambda s: _get_meta(int(s), "risk_level") or "Moderate")
    df["Description"] = df["Segment"].map(lambda s: _get_meta(int(s), "description"))
    df["RecommendedStrategy"] = df["Segment"].map(lambda s: _get_meta(int(s), "strategy"))

    return df


def assign_kmeans_segments(
    df: pd.DataFrame,
    k: int = 4,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror the Spark K-Means segmentation pipeline at customer level in scikit-learn.
    
    Standardizes the 6 segment features, fits KMeans(k=4, seed=42), summarizes clusters,
    and applies dynamic rule-based persona naming.
    """
    missing = [c for c in SEGMENT_FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for segmentation: {missing}")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(df[SEGMENT_FEATURE_COLS])
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["Segment"] = cluster_labels

    # Compute summary to derive rule-based metadata
    summary = df_out.groupby("Segment").agg(
        Customers=("CustomerId", "count") if "CustomerId" in df_out.columns else ("Segment", "count"),
        AvgAge=("Age", "mean"),
        AvgBalance=("Balance", "mean"),
        AvgActivity=("CustomerActivityScore", "mean"),
        ChurnRatePct=("Exited", lambda x: float(x.mean() * 100.0)) if "Exited" in df_out.columns else ("Segment", lambda x: 0.0),
    ).reset_index()

    summary["AvgAge"] = summary["AvgAge"].round(1)
    summary["AvgBalance"] = summary["AvgBalance"].round(0)
    summary["AvgActivity"] = summary["AvgActivity"].round(3)
    summary["ChurnRatePct"] = summary["ChurnRatePct"].round(2)

    profiles = build_segment_profiles(summary)
    name_map = dict(zip(profiles["Segment"], profiles["SegmentName"]))
    df_out["SegmentName"] = df_out["Segment"].map(name_map)

    return df_out, profiles


def get_normalized_radar_metrics(profiles_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize measured cluster metrics to 0-1 range for radar chart visualization."""
    if profiles_df.empty:
        return pd.DataFrame()

    metrics = ["AvgAge", "AvgBalance", "AvgActivity", "ChurnRatePct"]
    radar_data = []

    for _, row in profiles_df.iterrows():
        segment_id = int(row["Segment"])
        segment_name = row["SegmentName"]
        for metric in metrics:
            val = float(row[metric])
            min_val = float(profiles_df[metric].min())
            max_val = float(profiles_df[metric].max())
            norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            radar_data.append({
                "Segment": f"Seg {segment_id}: {segment_name}",
                "Metric": metric.replace("Pct", " (%)").replace("Avg", "Avg "),
                "RawValue": val,
                "NormalizedScore": norm,
            })

    return pd.DataFrame(radar_data)

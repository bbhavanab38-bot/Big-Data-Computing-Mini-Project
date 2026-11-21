"""Pure-Python Micro-Batch Streaming Fallback Engine.

Consumes simulated event files from data/streaming/events, computes rolling customer features,
scores calibrated churn risk using src.features.score_customer_records, and writes
snapshots and heartbeats with engine="Pure-Python Micro-Batch Fallback".
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src import config
from src.features import score_customer_records

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = ROOT / "data" / "streaming" / "events"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "streaming" / "output"
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_FILE = DEFAULT_OUTPUT_DIR / "risk_snapshots.csv"
HEARTBEAT_FILE = DEFAULT_OUTPUT_DIR / "heartbeat.json"


def run_microbatch_job(
    input_dir: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    max_events: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a micro-batch cycle processing available event files."""
    start_time = time.time()
    in_path = Path(input_dir) if input_dir else DEFAULT_INPUT_DIR
    out_path = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    out_path.mkdir(parents=True, exist_ok=True)

    snapshot_path = out_path / "risk_snapshots.csv"
    heartbeat_path = out_path / "heartbeat.json"

    # Read all JSON/JSONL event files in directory
    event_files = sorted(list(in_path.glob("*.json")) + list(in_path.glob("*.jsonl")))
    if not event_files:
        heartbeat = {
            "engine": "Pure-Python Micro-Batch Fallback",
            "status": "IDLE",
            "last_batch_time": datetime.now(timezone.utc).isoformat(),
            "events_processed": 0,
            "customers_tracked": 0,
            "throughput_eps": 0.0,
            "source": "simulated",
            "message": "No event files detected in input directory.",
        }
        with open(heartbeat_path, "w", encoding="utf-8") as f:
            json.dump(heartbeat, f, indent=2)
        return heartbeat

    raw_events: List[Dict[str, Any]] = []
    for f in event_files:
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        raw_events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        if max_events and len(raw_events) >= max_events:
            raw_events = raw_events[:max_events]
            break

    if not raw_events:
        return {"status": "IDLE", "events_processed": 0}

    pdf = pd.DataFrame(raw_events)

    # Load static profiles
    static_df = pd.read_csv(config.SCORED_DATASET_FILE)
    static_cols = [
        "CustomerId", "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "EstimatedSalary", "TransactionCount", "OnlineBankingUsage", "CustomerComplaints",
        "LoanAmount", "HasCrCard", "IsActiveMember", "Geography", "Gender", "LoanStatus",
        "CalibratedProbability"
    ]
    static_sub = static_df[[c for c in static_cols if c in static_df.columns]].copy()
    static_sub = static_sub.rename(columns={"CalibratedProbability": "BaselineProbability"})

    # Extract event fields
    def _extract_bal(p):
        if isinstance(p, dict) and "new_balance" in p:
            return float(p["new_balance"])
        return None

    pdf["new_balance"] = pdf["payload"].map(_extract_bal)
    pdf["is_complaint"] = (pdf["event_type"] == "complaint").astype(int)
    pdf["is_tx"] = (pdf["event_type"] == "transaction").astype(int)

    # Customer aggregates
    cust_aggs = pdf.groupby("customer_id").agg(
        event_count=("event_id", "count"),
        window_tx_count=("is_tx", "sum"),
        window_complaints=("is_complaint", "sum"),
        latest_balance=("new_balance", "last"),
    ).reset_index().rename(columns={"customer_id": "CustomerId"})

    merged = pd.merge(static_sub, cust_aggs, on="CustomerId", how="inner")
    if merged.empty:
        return {"status": "NO_MATCH", "events_processed": len(pdf)}

    updated_records = []
    for _, row in merged.iterrows():
        rec = row.to_dict()
        if pd.notna(row.get("latest_balance")):
            rec["Balance"] = float(row["latest_balance"])
        rec["CustomerComplaints"] = float(row["CustomerComplaints"]) + float(row["window_complaints"])
        
        # At-risk behavior: lack of transactions + severe balance drop
        if row["window_tx_count"] == 0 and pd.notna(row.get("latest_balance")) and row["latest_balance"] < (row["Balance"] * 0.5):
            rec["IsActiveMember"] = 0
            rec["TransactionCount"] = max(0, int(row["TransactionCount"]) // 2)

        updated_records.append(rec)

    eval_df = pd.DataFrame(updated_records)
    stream_probs = score_customer_records(eval_df)
    eval_df["StreamProbability"] = stream_probs
    eval_df["DeltaRisk"] = eval_df["StreamProbability"] - eval_df["BaselineProbability"]
    eval_df["RiskCategory"] = eval_df["StreamProbability"].map(config.get_risk_category)
    eval_df["ProcessedAt"] = datetime.now(timezone.utc).isoformat()
    eval_df["Engine"] = "Pure-Python Micro-Batch Fallback"

    out_cols = [
        "CustomerId", "ProcessedAt", "BaselineProbability", "StreamProbability",
        "DeltaRisk", "RiskCategory", "event_count", "window_tx_count",
        "window_complaints", "Balance", "Engine"
    ]
    snapshot_sub = eval_df[out_cols].copy()

    # Save snapshots
    if snapshot_path.exists():
        existing = pd.read_csv(snapshot_path)
        combined = pd.concat([existing[~existing["CustomerId"].isin(snapshot_sub["CustomerId"])], snapshot_sub], ignore_index=True)
        combined.to_csv(snapshot_path, index=False)
    else:
        snapshot_sub.to_csv(snapshot_path, index=False)

    duration = max(0.001, time.time() - start_time)
    throughput = len(pdf) / duration
    heartbeat_data = {
        "engine": "Pure-Python Micro-Batch Fallback",
        "status": "RUNNING",
        "last_batch_time": datetime.now(timezone.utc).isoformat(),
        "events_processed": len(pdf),
        "customers_tracked": len(snapshot_sub),
        "throughput_eps": round(throughput, 1),
        "batch_duration_sec": round(duration, 3),
        "source": "simulated",
    }

    with open(heartbeat_path, "w", encoding="utf-8") as f:
        json.dump(heartbeat_data, f, indent=2)

    return heartbeat_data


if __name__ == "__main__":
    res = run_microbatch_job()
    print("[Micro-Batch Runner] Completed cycle:")
    print(json.dumps(res, indent=2))

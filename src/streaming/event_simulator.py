"""Simulated Banking Event Stream Generator.

Produces synthetic timestamped banking events (transactions, logins, complaints, balance changes)
for existing customers to drive Spark Structured Streaming and micro-batch risk monitoring.
Includes a configurable scenario where a designated cohort displays declining engagement and
balance drawdowns, allowing real-time risk escalation to be monitored.

DISCLAIMER: All events, timestamps, and customer behaviors generated here are strictly synthetic
and simulated. No real banking events or customer telemetry exist in this project.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src import config

# Default directories
DEFAULT_EVENT_DIR = Path(__file__).resolve().parents[2] / "data" / "streaming" / "events"


def generate_simulated_stream_events(
    customers_df: Optional[pd.DataFrame] = None,
    num_events: int = 1000,
    at_risk_ratio: float = 0.20,
    window_days: int = 14,
    start_time: Optional[datetime] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a sequence of synthetic banking events with realistic temporal patterns.
    
    Args:
        customers_df: DataFrame of existing bank customers. If None, loaded from scored_customers.csv.
        num_events: Target number of events to synthesize.
        at_risk_ratio: Fraction of customers assigned to the declining engagement / risk-escalation cohort.
        window_days: Span of historical days over which to distribute events.
        start_time: Reference starting timestamp. Defaults to UTC now minus window_days.
        seed: Random seed for strict reproducibility.
        
    Returns:
        List of event dictionaries ready for JSON serialization.
    """
    rng = np.random.RandomState(seed)

    if customers_df is None:
        customers_df = pd.read_csv(config.SCORED_DATASET_FILE)

    if customers_df.empty:
        raise ValueError("Customer dataset is empty. Cannot generate events.")

    # Select candidate customer sample (e.g. up to 200 unique customers for focused monitoring)
    sample_size = min(200, len(customers_df))
    sample_customers = customers_df.sample(n=sample_size, random_state=seed).copy().reset_index(drop=True)

    # Assign deteriorating at-risk cohort
    num_at_risk = int(sample_size * at_risk_ratio)
    at_risk_indices = set(rng.choice(sample_size, size=num_at_risk, replace=False))
    sample_customers["is_deteriorating"] = [i in at_risk_indices for i in range(sample_size)]

    if start_time is None:
        ref_end = datetime.now(timezone.utc)
        ref_start = ref_end - timedelta(days=window_days)
    else:
        ref_start = start_time
        ref_end = ref_start + timedelta(days=window_days)

    total_seconds = int((ref_end - ref_start).total_seconds())

    events: List[Dict[str, Any]] = []
    merchants = ["Supermarket", "Coffee Shop", "Pharmacy", "Electronics Store", "Utility Co", "App Store", "Airline"]
    complaint_cats = ["fee_dispute", "card_declined", "unresponsive_support", "interest_rate_inquiry"]

    # Generate events
    for evt_idx in range(num_events):
        # Pick customer
        cust_row = sample_customers.sample(n=1, random_state=rng.randint(0, 1000000)).iloc[0]
        cid = int(cust_row["CustomerId"])
        is_det = bool(cust_row["is_deteriorating"])
        base_balance = float(cust_row.get("Balance", 50000.0))

        # Random timestamp within window
        sec_offset = int(rng.uniform(0, total_seconds))
        evt_time = ref_start + timedelta(seconds=sec_offset)
        is_second_half = sec_offset > (total_seconds // 2)

        # Deteriorating customers in the second half of the window experience declining health
        if is_det and is_second_half:
            # Shift distribution towards balance drawdown and complaints
            evt_type = rng.choice(
                ["balance_change", "complaint", "login", "transaction"],
                p=[0.45, 0.35, 0.15, 0.05],  # Transactions drop dramatically, complaints surge
            )
        else:
            # Normal distribution
            evt_type = rng.choice(
                ["transaction", "login", "balance_change", "complaint"],
                p=[0.60, 0.25, 0.14, 0.01],
            )

        payload: Dict[str, Any] = {}

        if evt_type == "transaction":
            amount = round(float(rng.uniform(5.0, 350.0)), 2)
            merchant = str(rng.choice(merchants))
            direction = "debit" if rng.uniform() > 0.15 else "credit"
            payload = {
                "amount": amount,
                "merchant": merchant,
                "direction": direction,
                "channel": "pos" if rng.uniform() > 0.4 else "online",
            }
        elif evt_type == "login":
            payload = {
                "channel": "mobile_app" if rng.uniform() > 0.3 else "web_portal",
                "session_duration_sec": int(rng.uniform(15, 300)),
            }
        elif evt_type == "complaint":
            payload = {
                "category": str(rng.choice(complaint_cats)),
                "severity": "high" if is_det else "low",
                "channel": "call_center",
            }
        elif evt_type == "balance_change":
            if is_det and is_second_half:
                # Severe balance drawdown: 40% to 75% reduction
                drawdown_pct = float(rng.uniform(0.40, 0.75))
                change_amount = -round(base_balance * drawdown_pct, 2)
                new_bal = max(0.0, base_balance + change_amount)
                reason = "external_wire_outflow"
            else:
                change_amount = round(float(rng.uniform(-300.0, 600.0)), 2)
                new_bal = max(0.0, base_balance + change_amount)
                reason = "normal_bill_or_deposit"

            payload = {
                "change_amount": change_amount,
                "new_balance": new_bal,
                "reason": reason,
            }

        event_record = {
            "event_id": f"evt_{seed}_{evt_idx:06d}",
            "customer_id": cid,
            "timestamp": evt_time.isoformat(),
            "event_type": evt_type,
            "payload": payload,
            "source": "simulated",
            "is_deteriorating_cohort": is_det,
        }
        events.append(event_record)

    # Sort events chronologically
    events.sort(key=lambda x: x["timestamp"])
    return events


def write_events_to_directory(
    events: List[Dict[str, Any]],
    output_dir: Path | str,
    batch_size: int = 100,
) -> List[Path]:
    """Write events into partitioned JSONL files for streaming consumption."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written_files: List[Path] = []
    total_events = len(events)

    for i in range(0, total_events, batch_size):
        chunk = events[i : i + batch_size]
        batch_id = i // batch_size
        file_path = out_path / f"events_batch_{batch_id:04d}.json"
        
        with open(file_path, "w", encoding="utf-8") as f:
            for evt in chunk:
                f.write(json.dumps(evt) + "\n")
        written_files.append(file_path)

    return written_files


def run_event_generator(
    events: int = 1000,
    output_dir: Optional[str] = None,
    at_risk_ratio: float = 0.20,
    seed: int = 42,
) -> int:
    """CLI/API entrypoint to generate and write simulated events."""
    target_dir = Path(output_dir) if output_dir else DEFAULT_EVENT_DIR
    print(f"[Streaming Generator] Synthesizing {events:,} events (at_risk_ratio={at_risk_ratio:.0%}, seed={seed})...")
    evt_list = generate_simulated_stream_events(
        num_events=events,
        at_risk_ratio=at_risk_ratio,
        seed=seed,
    )
    files = write_events_to_directory(evt_list, target_dir)
    print(f"[Streaming Generator] Successfully wrote {len(evt_list):,} events into {len(files)} JSON files at {target_dir}")
    return len(evt_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated Banking Event Generator for Real-Time Streaming")
    parser.add_argument("--events", type=int, default=1000, help="Number of simulated events to generate")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_EVENT_DIR), help="Output directory for events")
    parser.add_argument("--at-risk-ratio", type=float, default=0.20, help="Ratio of customers displaying declining activity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    run_event_generator(
        events=args.events,
        output_dir=args.output_dir,
        at_risk_ratio=args.at_risk_ratio,
        seed=args.seed,
    )

"""Spark Structured Streaming Risk Monitoring Job.

Consumes streaming synthetic banking events from file source, applies event-time windowing
with watermarks, joins static customer profiles, computes rolling feature deltas, scores
calibrated churn risk with the registered model, and writes risk snapshots and heartbeats.

Usage:
    spark-submit spark/spark_streaming_risk.py
    # or with Python:
    python spark/spark_streaming_risk.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Guarantee matching Python interpreter across Spark driver and executors
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Ensure root is in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from src import config
from src.features import score_customer_records, build_feature_dataframe

# Directory configurations
STREAM_INPUT_DIR = os.getenv("STREAM_INPUT_DIR", str(ROOT / "data" / "streaming" / "events"))
STREAM_OUTPUT_DIR = os.getenv("STREAM_OUTPUT_DIR", str(ROOT / "data" / "streaming" / "output"))
STREAM_CHECKPOINT_DIR = os.getenv("STREAM_CHECKPOINT_DIR", str(ROOT / "data" / "streaming" / "checkpoints"))
HEARTBEAT_FILE = Path(STREAM_OUTPUT_DIR) / "heartbeat.json"
SNAPSHOT_FILE = Path(STREAM_OUTPUT_DIR) / "risk_snapshots.csv"
SPARK_MASTER = os.getenv("SPARK_MASTER", "local[*]")


def process_microbatch(batch_df, batch_id: int):
    """Micro-batch processor function executed by Spark Structured Streaming."""
    batch_start = time.time()
    pdf = batch_df.toPandas()
    
    if pdf.empty:
        return

    # Load static customer baseline profiles
    static_df = pd.read_csv(config.SCORED_DATASET_FILE)
    static_cols = [
        "CustomerId", "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "EstimatedSalary", "TransactionCount", "OnlineBankingUsage", "CustomerComplaints",
        "LoanAmount", "HasCrCard", "IsActiveMember", "Geography", "Gender", "LoanStatus",
        "CalibratedProbability"
    ]
    static_sub = static_df[[c for c in static_cols if c in static_df.columns]].copy()
    static_sub = static_sub.rename(columns={"CalibratedProbability": "BaselineProbability"})

    # Extract event payload fields
    def _extract_balance(row):
        payload = row.get("payload")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                payload = {}
        if isinstance(payload, dict) and "new_balance" in payload:
            return float(payload["new_balance"])
        return None

    def _extract_complaint(row):
        return 1 if row.get("event_type") == "complaint" else 0

    def _extract_tx(row):
        return 1 if row.get("event_type") == "transaction" else 0

    pdf["new_balance"] = pdf.apply(_extract_balance, axis=1)
    pdf["is_complaint"] = pdf.apply(_extract_complaint, axis=1)
    pdf["is_tx"] = pdf.apply(_extract_tx, axis=1)

    # Aggregate events per customer in this microbatch
    cust_aggs = pdf.groupby("customer_id").agg(
        event_count=("event_id", "count"),
        window_tx_count=("is_tx", "sum"),
        window_complaints=("is_complaint", "sum"),
        latest_balance=("new_balance", "last"),
    ).reset_index().rename(columns={"customer_id": "CustomerId"})

    # Join with static profiles
    merged = pd.merge(static_sub, cust_aggs, on="CustomerId", how="inner")
    if merged.empty:
        return

    # Update dynamic features based on stream window
    updated_records = []
    for _, row in merged.iterrows():
        rec = row.to_dict()
        # If new balance was observed, update balance
        if pd.notna(row.get("latest_balance")):
            rec["Balance"] = float(row["latest_balance"])
        # Update complaints count
        rec["CustomerComplaints"] = float(row["CustomerComplaints"]) + float(row["window_complaints"])
        # If transactions dropped to 0 and balance dropped significantly, mark inactive
        if row["window_tx_count"] == 0 and pd.notna(row.get("latest_balance")) and row["latest_balance"] < (row["Balance"] * 0.5):
            rec["IsActiveMember"] = 0
            rec["TransactionCount"] = max(0, int(row["TransactionCount"]) // 2)
        updated_records.append(rec)

    eval_df = pd.DataFrame(updated_records)
    # Score using unified feature module (guaranteed zero skew)
    stream_probs = score_customer_records(eval_df)
    eval_df["StreamProbability"] = stream_probs
    eval_df["DeltaRisk"] = eval_df["StreamProbability"] - eval_df["BaselineProbability"]
    eval_df["RiskCategory"] = eval_df["StreamProbability"].map(config.get_risk_category)
    eval_df["ProcessedAt"] = datetime.now(timezone.utc).isoformat()
    eval_df["Engine"] = "PySpark Structured Streaming"

    # Save risk snapshots (append or update)
    Path(STREAM_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    out_cols = [
        "CustomerId", "ProcessedAt", "BaselineProbability", "StreamProbability",
        "DeltaRisk", "RiskCategory", "event_count", "window_tx_count",
        "window_complaints", "Balance", "Engine"
    ]
    snapshot_sub = eval_df[out_cols].copy()
    
    if SNAPSHOT_FILE.exists():
        existing = pd.read_csv(SNAPSHOT_FILE)
        # Update existing records or append new
        combined = pd.concat([existing[~existing["CustomerId"].isin(snapshot_sub["CustomerId"])], snapshot_sub], ignore_index=True)
        combined.to_csv(SNAPSHOT_FILE, index=False)
    else:
        snapshot_sub.to_csv(SNAPSHOT_FILE, index=False)

    # Write Heartbeat
    duration = max(0.001, time.time() - batch_start)
    throughput = len(pdf) / duration
    heartbeat_data = {
        "engine": "PySpark Structured Streaming",
        "status": "RUNNING",
        "last_batch_id": batch_id,
        "last_batch_time": datetime.now(timezone.utc).isoformat(),
        "events_processed": len(pdf),
        "customers_tracked": len(snapshot_sub),
        "throughput_eps": round(throughput, 1),
        "batch_duration_sec": round(duration, 3),
        "checkpoint_dir": STREAM_CHECKPOINT_DIR,
        "source": "simulated",
    }
    with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
        json.dump(heartbeat_data, f, indent=2)

    print(f"[Spark Structured Streaming] Micro-batch {batch_id}: {len(pdf)} events, {len(snapshot_sub)} customer scores updated in {duration:.2f}s ({throughput:.1f} eps)")


def run_spark_streaming():
    """Initialize and run Spark Structured Streaming file stream."""
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql import types as T
    except ImportError:
        print("[Streaming Error] PySpark is not installed. Please use the pure-Python fallback engine.")
        sys.exit(1)

    print(f"[Spark Structured Streaming] Starting job with Master='{SPARK_MASTER}'...")
    print(f"[Spark Structured Streaming] Monitoring input directory: {STREAM_INPUT_DIR}")

    Path(STREAM_INPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(STREAM_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(STREAM_CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder.appName("BankChurnStructuredStreamingRiskMonitor")
        .master(SPARK_MASTER)
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Define schema for simulated JSON events
    event_schema = T.StructType([
        T.StructField("event_id", T.StringType(), True),
        T.StructField("customer_id", T.LongType(), True),
        T.StructField("timestamp", T.StringType(), True),
        T.StructField("event_type", T.StringType(), True),
        T.StructField("payload", T.StringType(), True),
        T.StructField("source", T.StringType(), True),
        T.StructField("is_deteriorating_cohort", T.BooleanType(), True),
    ])

    raw_stream = (
        spark.readStream.format("json")
        .schema(event_schema)
        .option("maxFilesPerTrigger", 5)
        .load(STREAM_INPUT_DIR)
    )

    # Cast timestamp and apply watermark
    parsed_stream = raw_stream.withColumn("event_timestamp", F.to_timestamp(F.col("timestamp"))).withWatermark("event_timestamp", "1 hour")

    query = (
        parsed_stream.writeStream.foreachBatch(process_microbatch)
        .trigger(processingTime="5 seconds")
        .option("checkpointLocation", STREAM_CHECKPOINT_DIR)
        .start()
    )

    print("[Spark Structured Streaming] Query started successfully. Listening for event batches (Press Ctrl+C to stop)...")
    try:
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[Spark Structured Streaming] Termination requested. Stopping stream query...")
        query.stop()
        spark.stop()
        print("[Spark Structured Streaming] Stopped cleanly.")


if __name__ == "__main__":
    run_spark_streaming()

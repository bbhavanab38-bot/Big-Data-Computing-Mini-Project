"""Dashboard interface for Real-Time Streaming and Risk Engine status."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import pandas as pd

from src.streaming.microbatch_runner import (
    DEFAULT_INPUT_DIR,
    HEARTBEAT_FILE,
    SNAPSHOT_FILE,
    run_microbatch_job,
)


def get_stream_heartbeat() -> Optional[Dict[str, Any]]:
    """Load latest streaming engine heartbeat metadata."""
    if not HEARTBEAT_FILE.exists():
        return None
    try:
        with open(HEARTBEAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_risk_snapshots() -> pd.DataFrame:
    """Load latest risk snapshots for streaming-tracked customers."""
    if not SNAPSHOT_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(SNAPSHOT_FILE)
    except Exception:
        return pd.DataFrame()


def load_recent_stream_events(limit: int = 100) -> pd.DataFrame:
    """Read recent events from the input streaming directory."""
    if not DEFAULT_INPUT_DIR.exists():
        return pd.DataFrame()

    files = sorted(list(DEFAULT_INPUT_DIR.glob("*.json")) + list(DEFAULT_INPUT_DIR.glob("*.jsonl")), reverse=True)
    if not files:
        return pd.DataFrame()

    events: List[Dict[str, Any]] = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
                        if len(events) >= limit:
                            break
        except Exception:
            continue
        if len(events) >= limit:
            break

    if not events:
        return pd.DataFrame()

    df = pd.DataFrame(events)
    # Sort descending by timestamp
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)
    return df


def trigger_microbatch_cycle() -> Dict[str, Any]:
    """Execute a single micro-batch run on available event files."""
    return run_microbatch_job()

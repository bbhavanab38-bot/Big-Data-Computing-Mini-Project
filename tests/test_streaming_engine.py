"""Unit tests for the Streaming Risk Engine and Micro-Batch Fallback Runner."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import pandas as pd

from src.streaming.microbatch_runner import run_microbatch_job


class TestStreamingEngine(unittest.TestCase):
    def setUp(self):
        self.temp_in = Path(tempfile.mkdtemp())
        self.temp_out = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_in, ignore_errors=True)
        shutil.rmtree(self.temp_out, ignore_errors=True)

    def test_idle_heartbeat_on_empty_directory(self):
        """Test engine writes clean IDLE heartbeat when no events are available."""
        hb = run_microbatch_job(input_dir=self.temp_in, output_dir=self.temp_out)
        self.assertEqual(hb["status"], "IDLE")
        self.assertEqual(hb["events_processed"], 0)
        self.assertEqual(hb["engine"], "Pure-Python Micro-Batch Fallback")

        hb_file = self.temp_out / "heartbeat.json"
        self.assertTrue(hb_file.exists())
        with open(hb_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
            self.assertEqual(saved["status"], "IDLE")

    def test_microbatch_processing_and_risk_scoring(self):
        """Test micro-batch processes hand-crafted events and updates customer risk score."""
        # Create synthetic events for real customer 15601539
        # 1 complaint and severe balance drawdown to $1,000 from original balance
        events = [
            {
                "event_id": "test_evt_001",
                "customer_id": 15601539,
                "timestamp": "2026-11-15T12:00:00Z",
                "event_type": "complaint",
                "payload": {"category": "fee_dispute", "severity": "high"},
                "source": "simulated",
            },
            {
                "event_id": "test_evt_002",
                "customer_id": 15601539,
                "timestamp": "2026-11-15T12:05:00Z",
                "event_type": "balance_change",
                "payload": {"new_balance": 500.0, "change_amount": -45000.0, "reason": "outflow"},
                "source": "simulated",
            },
        ]

        event_file = self.temp_in / "events_test.json"
        with open(event_file, "w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        # Execute micro-batch
        hb = run_microbatch_job(input_dir=self.temp_in, output_dir=self.temp_out)
        self.assertEqual(hb["status"], "RUNNING")
        self.assertEqual(hb["events_processed"], 2)
        self.assertEqual(hb["customers_tracked"], 1)

        # Check snapshot output
        snap_file = self.temp_out / "risk_snapshots.csv"
        self.assertTrue(snap_file.exists())

        snap_df = pd.read_csv(snap_file)
        self.assertEqual(len(snap_df), 1)
        cust_row = snap_df.iloc[0]
        self.assertEqual(int(cust_row["CustomerId"]), 15601539)
        self.assertIn("BaselineProbability", snap_df.columns)
        self.assertIn("StreamProbability", snap_df.columns)
        self.assertIn("DeltaRisk", snap_df.columns)
        self.assertIn("Engine", snap_df.columns)
        self.assertEqual(cust_row["Engine"], "Pure-Python Micro-Batch Fallback")
        self.assertGreaterEqual(float(cust_row["StreamProbability"]), 0.0)
        self.assertLessEqual(float(cust_row["StreamProbability"]), 1.0)


if __name__ == "__main__":
    unittest.main()

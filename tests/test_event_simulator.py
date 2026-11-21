"""Unit tests for the Simulated Banking Event Generator (src/streaming/event_simulator.py)."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.streaming.event_simulator import (
    generate_simulated_stream_events,
    write_events_to_directory,
    run_event_generator,
)


class TestEventSimulator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_event_generation_schema_and_counts(self):
        """Test event generation produces correct counts and conforms to JSON schema."""
        events = generate_simulated_stream_events(num_events=50, at_risk_ratio=0.20, seed=42)
        self.assertEqual(len(events), 50)

        for evt in events:
            self.assertIn("event_id", evt)
            self.assertIn("customer_id", evt)
            self.assertIn("timestamp", evt)
            self.assertIn("event_type", evt)
            self.assertIn("payload", evt)
            self.assertEqual(evt["source"], "simulated")
            self.assertIn(evt["event_type"], {"transaction", "login", "complaint", "balance_change"})

    def test_event_generation_reproducibility(self):
        """Test fixed seed produces identical events sequence."""
        events_1 = generate_simulated_stream_events(num_events=30, seed=999)
        events_2 = generate_simulated_stream_events(num_events=30, seed=999)

        self.assertEqual(len(events_1), len(events_2))
        for e1, e2 in zip(events_1, events_2):
            self.assertEqual(e1["event_id"], e2["event_id"])
            self.assertEqual(e1["customer_id"], e2["customer_id"])
            self.assertEqual(e1["event_type"], e2["event_type"])
            self.assertEqual(e1["payload"], e2["payload"])

    def test_deteriorating_cohort_behavior(self):
        """Verify deteriorating cohort produces balance drawdowns and complaints."""
        events = generate_simulated_stream_events(num_events=300, at_risk_ratio=0.50, seed=42)
        det_events = [e for e in events if e.get("is_deteriorating_cohort")]
        self.assertGreater(len(det_events), 0)

        # Check that complaints and balance changes exist in deteriorating events
        det_types = {e["event_type"] for e in det_events}
        self.assertTrue("complaint" in det_types or "balance_change" in det_types)

    def test_write_events_to_directory(self):
        """Verify events write into partitioned JSON files and parse cleanly."""
        events = generate_simulated_stream_events(num_events=150, seed=42)
        files = write_events_to_directory(events, self.temp_dir, batch_size=50)

        self.assertEqual(len(files), 3)
        read_count = 0
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                for line in fp:
                    evt = json.loads(line)
                    self.assertEqual(evt["source"], "simulated")
                    read_count += 1

        self.assertEqual(read_count, 150)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for Model Registry: versioning, metadata serialization, and environment validation."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.registry import (
    build_model_bundle,
    compute_data_hash,
    load_model_bundle,
    load_registry,
    save_to_registry,
)


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.dummy_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.dummy_pipe = LogisticRegression()

    def test_compute_data_hash_deterministic(self):
        hash1 = compute_data_hash(self.dummy_df)
        hash2 = compute_data_hash(self.dummy_df)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 12)

    def test_build_model_bundle_metadata(self):
        bundle = build_model_bundle(
            name="TestModel",
            pipeline=self.dummy_pipe,
            features=["A", "B"],
            defaults={"A": 1.0, "B": 4.0},
            metrics={"ROC-AUC": 0.75, "Brier": 0.12},
            calibration_method="sigmoid",
            data_hash="abc123def456",
        )
        self.assertEqual(bundle["name"], "TestModel")
        self.assertTrue(bundle["model_version"].startswith("v_"))
        self.assertIn("trained_at", bundle)
        self.assertIn("sklearn_version", bundle)
        self.assertEqual(bundle["calibration_method"], "sigmoid")
        self.assertEqual(bundle["training_data_hash"], "abc123def456")

    def test_version_mismatch_validation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "mismatched_model.pkl"
            bundle = build_model_bundle(
                name="FutureModel",
                pipeline=self.dummy_pipe,
                features=["A"],
                defaults={},
                metrics={},
            )
            # Forge an incompatible sklearn version
            bundle["sklearn_version"] = "99.0.0"

            import joblib
            joblib.dump(bundle, file_path)

            with self.assertRaises(RuntimeError) as ctx:
                load_model_bundle(file_path, enforce_version=True)
            self.assertIn("Version mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()


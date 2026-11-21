"""Model Registry: versioned artifact packaging, metadata management, and validation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import joblib
import sklearn
from src import config

REGISTRY_PATH = config.REGISTRY_FILE


def compute_data_hash(df) -> str:
    """Compute sha256 checksum of a dataframe's values and column names."""
    hasher = hashlib.sha256()
    hasher.update(str(df.shape).encode("utf-8"))
    hasher.update(",".join(df.columns).encode("utf-8"))
    hasher.update(df.iloc[: min(len(df), 500)].to_csv(index=False).encode("utf-8"))
    return hasher.hexdigest()[:12]


def build_model_bundle(
    name: str,
    pipeline: Any,
    features: list,
    defaults: dict,
    metrics: dict,
    calibration_method: str = "none",
    data_hash: str = "unknown",
) -> dict:
    """Create a standardized model bundle containing serializable metadata."""
    now_utc = datetime.now(timezone.utc)
    timestamp_str = now_utc.strftime("%Y%m%d_%H%M%S")
    hash_seed = hashlib.sha256(f"{name}_{timestamp_str}_{data_hash}".encode("utf-8")).hexdigest()[:7]
    version = f"v_{timestamp_str}_{hash_seed}"

    return {
        "name": name,
        "model_version": version,
        "trained_at": now_utc.isoformat(),
        "sklearn_version": sklearn.__version__,
        "features": features,
        "defaults": defaults,
        "calibration_method": calibration_method,
        "training_data_hash": data_hash,
        "metrics": metrics,
        "pipeline": pipeline,
    }


def save_to_registry(bundle: dict, file_path: Path, is_current: bool = True):
    """Save model bundle to disk and register version entry in registry.json."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, file_path)

    reg = load_registry()
    version = bundle.get("model_version", "v_unknown")
    entry = {
        "name": bundle.get("name"),
        "model_version": version,
        "trained_at": bundle.get("trained_at"),
        "sklearn_version": bundle.get("sklearn_version"),
        "calibration_method": bundle.get("calibration_method"),
        "training_data_hash": bundle.get("training_data_hash"),
        "metrics": bundle.get("metrics"),
        "file_name": file_path.name,
    }
    reg["models"][version] = entry
    if is_current:
        reg["current_version"] = version

    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


def load_registry() -> dict:
    """Load or initialize models/registry.json."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"current_version": None, "models": {}}


def load_model_bundle(path: Optional[Path] = None, enforce_version: bool = True) -> dict:
    """Load a model bundle from path or registry, verifying scikit-learn version compatibility."""
    if path is None:
        reg = load_registry()
        curr_ver = reg.get("current_version")
        if curr_ver and curr_ver in reg.get("models", {}):
            file_name = reg["models"][curr_ver].get("file_name", "best_model.pkl")
            path = config.MODELS_DIR / file_name
        else:
            path = config.BEST_MODEL_FILE

    if not Path(path).exists():
        raise FileNotFoundError(f"Model bundle not found at {path}")

    bundle = joblib.load(path)

    # Scikit-learn version validation
    saved_version = bundle.get("sklearn_version")
    if saved_version is not None and enforce_version:
        curr_major_minor = ".".join(sklearn.__version__.split(".")[:2])
        saved_major_minor = ".".join(saved_version.split(".")[:2])
        if curr_major_minor != saved_major_minor:
            raise RuntimeError(
                f"Version mismatch: Model was trained with scikit-learn {saved_version}, "
                f"but current runtime environment has scikit-learn {sklearn.__version__}. "
                f"Re-train the model or match library versions."
            )

    return bundle


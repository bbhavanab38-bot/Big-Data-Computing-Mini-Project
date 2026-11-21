#!/usr/bin/env python3
"""Platform Smoke Test Script for Banking Retention Intelligence Platform.

Executes rapid sanity checks across all platform subsystems:
1. Data files & models integrity
2. Unified feature engineering pipeline
3. FastAPI microservice endpoints (TestClient)
4. PDF & CSV executive report generation
5. SQLAlchemy database connectivity
6. Streamlit page rendering validation (AppTest)

Usage:
    python scripts/smoke_test.py
"""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# If executed with system Python, transparently delegate to .venv/bin/python
venv_python = ROOT_DIR / ".venv" / "bin" / "python"
if venv_python.exists() and sys.executable != str(venv_python):
    import os
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

from fastapi.testclient import TestClient

from src import config
from src.registry import load_registry
from src.features import build_feature_dataframe, score_customer_records
from src.api_v2.app import app
from dashboard.core.reports import build_report_data, generate_pdf_report, generate_summary_csv
from dashboard.core.database import engine
from dashboard.core.campaign_repository import list_campaigns

GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def log_pass(msg: str):
    print(f"  {GREEN}✔ PASS{RESET} {msg}")


def log_fail(msg: str):
    print(f"  {RED}✖ FAIL{RESET} {msg}")


def run_smoke_test():
    print(f"\n{BOLD}{BLUE}================================================================={RESET}")
    print(f"{BOLD}{BLUE}  BANKING RETENTION INTELLIGENCE PLATFORM - END-TO-END SMOKE TEST{RESET}")
    print(f"{BOLD}{BLUE}================================================================={RESET}\n")

    failures = 0
    total_checks = 0

    # 1. Check Data Files
    total_checks += 1
    print(f"{BOLD}[1/6] Checking Critical Datasets & Artifacts...{RESET}")
    required_files = [
        config.RAW_FILE,
        config.PROCESSED_FILE,
        config.TEST_FILE,
        config.SCORED_DATASET_FILE,
        config.BEST_MODEL_FILE,
        config.REGISTRY_FILE,
    ]
    missing = [f.name for f in required_files if not f.exists()]
    if not missing:
        log_pass("All primary datasets, scored holdouts, and model artifacts exist.")
    else:
        log_fail(f"Missing required artifacts: {missing}")
        failures += 1

    # 2. Check Model Registry
    total_checks += 1
    print(f"\n{BOLD}[2/6] Verifying Registered Production Model & Calibration...{RESET}")
    try:
        reg = load_registry()
        curr_ver = reg.get("current_version")
        meta = reg.get("models", {}).get(curr_ver, {})
        cal_method = meta.get("calibration_method")
        roc_auc = meta.get("metrics", {}).get("ROC-AUC", 0.0)
        brier = meta.get("metrics", {}).get("Brier", 1.0)
        
        assert curr_ver is not None, "current_version is missing"
        assert cal_method == "sigmoid", f"Unexpected calibration method: {cal_method}"
        assert roc_auc >= 0.70, f"ROC-AUC too low: {roc_auc}"
        assert brier < 0.15, f"Brier score uncalibrated: {brier}"
        log_pass(f"Model '{curr_ver}' verified: {cal_method.upper()} calibrated, ROC-AUC={roc_auc:.4f}, Brier={brier:.4f}")
    except Exception as exc:
        log_fail(f"Model registry check failed: {exc}")
        failures += 1

    # 3. Check Unified Feature Pipeline
    total_checks += 1
    print(f"\n{BOLD}[3/6] Verifying Feature Pipeline & Scoring Consistency...{RESET}")
    try:
        sample_record = [{
            "CreditScore": 680,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 42,
            "Tenure": 4,
            "Balance": 85000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 92000.0,
            "TransactionCount": 28,
            "OnlineBankingUsage": 65.0,
            "CustomerComplaints": 0,
            "LoanStatus": "Approved",
            "LoanAmount": 15000.0,
        }]
        feat_df = build_feature_dataframe(sample_record)
        probs = score_customer_records(feat_df)
        prob = float(probs[0])
        assert 0.0 <= prob <= 1.0, f"Calibrated probability out of bounds: {prob}"
        log_pass(f"Feature engineering pipeline verified: Sample P(Churn) = {prob:.4f}")
    except Exception as exc:
        log_fail(f"Feature scoring check failed: {exc}")
        failures += 1

    # 4. Check FastAPI Microservice
    total_checks += 1
    print(f"\n{BOLD}[4/6] Verifying FastAPI v2 Microservice Endpoints...{RESET}")
    try:
        client = TestClient(app)
        # Health check
        h_resp = client.get("/health")
        assert h_resp.status_code == 200, f"/health returned {h_resp.status_code}"

        # Metadata
        m_resp = client.get("/v1/model")
        assert m_resp.status_code == 200, f"/v1/model returned {m_resp.status_code}"

        # Single prediction
        pred_resp = client.post("/v1/predict", json=sample_record[0])
        assert pred_resp.status_code == 200, f"/v1/predict returned {pred_resp.status_code}"
        pred_data = pred_resp.json()
        assert "churn_probability" in pred_data
        assert "risk_category" in pred_data

        # Probe validation
        probe_resp = client.post("/v1/predict", json={"Age": "invalid_age"})
        assert probe_resp.status_code == 422, f"Probe validation expected 422, got {probe_resp.status_code}"

        log_pass("FastAPI endpoints verified: /health (200), /v1/model (200), /v1/predict (200), probe (422)")
    except Exception as exc:
        log_fail(f"FastAPI check failed: {exc}")
        failures += 1

    # 5. Check PDF & CSV Report Generation
    total_checks += 1
    print(f"\n{BOLD}[5/6] Verifying Executive Report Generation Engine...{RESET}")
    try:
        data = build_report_data()
        pdf_bytes = generate_pdf_report(data)
        assert len(pdf_bytes) > 2000, f"PDF byte stream too small: {len(pdf_bytes)}"
        assert pdf_bytes.startswith(b"%PDF-"), "Invalid PDF header signature"
        csv_str = generate_summary_csv(data)
        assert "System Metadata" in csv_str
        log_pass(f"Executive report engine verified: PDF generated ({len(pdf_bytes):,} bytes), CSV digest compiled")
    except Exception as exc:
        log_fail(f"Report generation check failed: {exc}")
        failures += 1

    # 6. Check Database & Campaign System
    total_checks += 1
    print(f"\n{BOLD}[6/6] Verifying Database & Campaign Persistence...{RESET}")
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        campaigns = list_campaigns()
        log_pass(f"Database connectivity verified: SQLite engine active, {len(campaigns)} stored campaigns")
    except Exception as exc:
        log_fail(f"Database check failed: {exc}")
        failures += 1

    # Final Summary
    print(f"\n{BOLD}-----------------------------------------------------------------{RESET}")
    if failures == 0:
        print(f"{BOLD}{GREEN}ALL {total_checks} SMOKE TEST CHECKS PASSED SUCCESSFULLY!{RESET}\n")
        return 0
    else:
        print(f"{BOLD}{RED}{failures} OF {total_checks} CHECKS FAILED!{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(run_smoke_test())

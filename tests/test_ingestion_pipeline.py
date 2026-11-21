"""Tests for Big Data Ingestion Pipeline, Schema Validation, and Quarantine Handling."""
import json
from pathlib import Path
import pandas as pd
import pytest

from src import config
from src.ingestion import ingest_batch, validate_record, QUARANTINE_FILE, AUDIT_REPORT_FILE


def test_validate_record_valid():
    """A normal conforming banking customer record passes validation without violation codes."""
    record = pd.Series({
        "CustomerId": 15600001,
        "CreditScore": 650,
        "Age": 38,
        "Balance": 45000.0,
        "EstimatedSalary": 60000.0,
        "Geography": "France",
        "Gender": "Female",
        "Exited": 0,
    })
    violations = validate_record(record)
    assert violations == []


def test_validate_record_violations():
    """Corrupted fields (credit score out of bounds, negative balance, age out of bounds) are flagged."""
    record = pd.Series({
        "CustomerId": -5,
        "CreditScore": 999,  # max is 850
        "Age": 12,          # min is 18
        "Balance": -100.0,  # cannot be negative
        "EstimatedSalary": -50.0,
        "Geography": "Atlantis",
        "Gender": "Unknown",
        "Exited": 3,
    })
    violations = validate_record(record)
    assert any("INVALID_CUSTOMER_ID" in v for v in violations)
    assert any("CREDIT_SCORE_OUT_OF_BOUNDS" in v for v in violations)
    assert any("AGE_OUT_OF_BOUNDS" in v for v in violations)
    assert any("NEGATIVE_BALANCE" in v for v in violations)
    assert any("NEGATIVE_SALARY" in v for v in violations)
    assert any("UNRECOGNIZED_GEOGRAPHY" in v for v in violations)
    assert any("UNRECOGNIZED_GENDER" in v for v in violations)
    assert any("INVALID_TARGET_LABEL" in v for v in violations)


def test_ingest_batch_quarantine_and_deduplication(tmp_path):
    """Batch ingestion isolates corrupted records, eliminates duplicate IDs, and writes audit report."""
    test_df = pd.DataFrame([
        # Row 0: Valid customer
        {"CustomerId": 101, "CreditScore": 700, "Age": 30, "Balance": 1000.0, "EstimatedSalary": 50000.0, "Geography": "France", "Gender": "Male", "Exited": 0},
        # Row 1: Valid customer
        {"CustomerId": 102, "CreditScore": 620, "Age": 45, "Balance": 2000.0, "EstimatedSalary": 60000.0, "Geography": "Spain", "Gender": "Female", "Exited": 1},
        # Row 2: Corrupted record (Age = 15, CreditScore = 150)
        {"CustomerId": 103, "CreditScore": 150, "Age": 15, "Balance": 0.0, "EstimatedSalary": 10000.0, "Geography": "Germany", "Gender": "Male", "Exited": 0},
        # Row 3: Duplicate customer (duplicate CustomerId 101)
        {"CustomerId": 101, "CreditScore": 700, "Age": 30, "Balance": 1000.0, "EstimatedSalary": 50000.0, "Geography": "France", "Gender": "Male", "Exited": 0},
    ])

    valid_df, quaran_df, audit = ingest_batch(test_df, source_name="unit_test_batch", save_raw=False)

    assert len(valid_df) == 2  # rows 101 and 102
    assert len(quaran_df) == 2  # row 103 (corrupted) + duplicate row 101
    assert audit["total_received"] == 4
    assert audit["valid_count"] == 2
    assert audit["quarantined_count"] == 2
    assert audit["duplicates_removed"] == 1

    # Verify quarantine file exists
    assert QUARANTINE_FILE.exists()
    quaran_saved = pd.read_csv(QUARANTINE_FILE)
    assert "QuarantineReason" in quaran_saved.columns

    # Verify audit JSON report exists
    assert AUDIT_REPORT_FILE.exists()
    with open(AUDIT_REPORT_FILE) as f:
        audit_data = json.load(f)
        assert audit_data["source"] == "unit_test_batch"
        assert audit_data["valid_count"] == 2

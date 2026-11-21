"""Phase 1 - Big Data Ingestion Pipeline.

Validates incoming batch datasets against banking domain schemas and constraints,
deduplicates customer records, quarantines corrupted or out-of-bound rows with reason codes,
stores raw data separately from processed data, and generates an ingestion audit report.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from src import config

logger = logging.getLogger("banking_ingestion")

QUARANTINE_DIR = config.ROOT / "data" / "quarantine"
QUARANTINE_FILE = QUARANTINE_DIR / "corrupted_records.csv"
AUDIT_REPORT_FILE = config.REPORTS_DIR / "ingestion_audit.json"

# Domain validation boundaries
VALID_GEOGRAPHIES = {"France", "Germany", "Spain"}
VALID_GENDERS = {"Male", "Female"}
VALID_LOAN_STATUSES = {"Active", "Closed", "Defaulted", "None"}
MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850
MIN_AGE = 18
MAX_AGE = 100


def validate_record(row: pd.Series) -> List[str]:
    """Validate a single record against banking domain rules; returns list of violation codes."""
    violations = []

    # 1. CustomerId
    cid = row.get("CustomerId")
    if pd.isna(cid):
        violations.append("MISSING_CUSTOMER_ID")
    else:
        try:
            cid_int = int(cid)
            if cid_int <= 0:
                violations.append("INVALID_CUSTOMER_ID_NON_POSITIVE")
        except (ValueError, TypeError):
            violations.append("NON_NUMERIC_CUSTOMER_ID")

    # 2. Credit Score
    cs = row.get("CreditScore")
    if pd.notna(cs):
        try:
            cs_float = float(cs)
            if cs_float < MIN_CREDIT_SCORE or cs_float > MAX_CREDIT_SCORE:
                violations.append(f"CREDIT_SCORE_OUT_OF_BOUNDS_{cs_float}")
        except (ValueError, TypeError):
            violations.append("NON_NUMERIC_CREDIT_SCORE")

    # 3. Age
    age = row.get("Age")
    if pd.notna(age):
        try:
            age_float = float(age)
            if age_float < MIN_AGE or age_float > MAX_AGE:
                violations.append(f"AGE_OUT_OF_BOUNDS_{age_float}")
        except (ValueError, TypeError):
            violations.append("NON_NUMERIC_AGE")

    # 4. Balance
    bal = row.get("Balance")
    if pd.notna(bal):
        try:
            bal_float = float(bal)
            if bal_float < 0:
                violations.append("NEGATIVE_BALANCE")
        except (ValueError, TypeError):
            violations.append("NON_NUMERIC_BALANCE")

    # 5. Salary
    salary = row.get("EstimatedSalary")
    if pd.notna(salary):
        try:
            salary_float = float(salary)
            if salary_float < 0:
                violations.append("NEGATIVE_SALARY")
        except (ValueError, TypeError):
            violations.append("NON_NUMERIC_SALARY")

    # 6. Geography
    geo = row.get("Geography")
    if pd.notna(geo) and str(geo).strip() not in VALID_GEOGRAPHIES:
        violations.append(f"UNRECOGNIZED_GEOGRAPHY_{geo}")

    # 7. Gender
    gender = row.get("Gender")
    if pd.notna(gender) and str(gender).strip() not in VALID_GENDERS:
        violations.append(f"UNRECOGNIZED_GENDER_{gender}")

    # 8. Target Label Exited (if present in supervised training batch)
    if "Exited" in row and pd.notna(row["Exited"]):
        try:
            ex_val = int(row["Exited"])
            if ex_val not in (0, 1):
                violations.append("INVALID_TARGET_LABEL")
        except (ValueError, TypeError):
            violations.append("NON_INTEGER_TARGET_LABEL")

    return violations


def ingest_batch(
    source_df: pd.DataFrame,
    source_name: str = "batch_csv",
    save_raw: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute ingestion pipeline: validate, quarantine invalid records, deduplicate, and log audit."""
    start_time = datetime.now(timezone.utc)
    total_records = len(source_df)
    logger.info("Starting ingestion of %d records from source: %s", total_records, source_name)

    df = source_df.copy()
    quarantined_rows: List[Dict[str, Any]] = []
    valid_indices: List[int] = []

    # Record-by-record schema validation
    for idx, row in df.iterrows():
        reasons = validate_record(row)
        if reasons:
            rec = row.to_dict()
            rec["QuarantineReason"] = "; ".join(reasons)
            rec["QuarantinedAt"] = datetime.now(timezone.utc).isoformat()
            quarantined_rows.append(rec)
        else:
            valid_indices.append(idx)

    valid_df = df.loc[valid_indices].copy()

    # Deduplication on CustomerId
    duplicates_count = 0
    if "CustomerId" in valid_df.columns:
        initial_valid = len(valid_df)
        dup_mask = valid_df.duplicated(subset=["CustomerId"], keep="first")
        duplicates_count = int(dup_mask.sum())
        if duplicates_count > 0:
            for _, dup_row in valid_df[dup_mask].iterrows():
                rec = dup_row.to_dict()
                rec["QuarantineReason"] = "DUPLICATE_CUSTOMER_ID"
                rec["QuarantinedAt"] = datetime.now(timezone.utc).isoformat()
                quarantined_rows.append(rec)
            valid_df = valid_df[~dup_mask].copy()

    # Create quarantine DataFrame
    quarantine_df = pd.DataFrame(quarantined_rows)
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    if not quarantine_df.empty:
        quarantine_df.to_csv(QUARANTINE_FILE, index=False)
        logger.warning("Quarantined %d records to %s", len(quarantine_df), QUARANTINE_FILE)
    elif QUARANTINE_FILE.exists():
        # Clean previous quarantine if empty batch
        pass

    # Save raw valid batch if requested
    if save_raw and not valid_df.empty:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        valid_df.to_csv(config.RAW_FILE, index=False)

    # Ingestion audit statistics
    audit_summary = {
        "source": source_name,
        "ingested_at": start_time.isoformat(),
        "total_received": total_records,
        "valid_count": len(valid_df),
        "quarantined_count": len(quarantine_df),
        "duplicates_removed": duplicates_count,
        "quarantine_file": str(QUARANTINE_FILE) if not quarantine_df.empty else None,
        "data_columns": list(valid_df.columns),
    }

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_REPORT_FILE, "w") as af:
        json.dump(audit_summary, af, indent=2)

    logger.info(
        "Ingestion completed: %d valid records, %d quarantined, %d duplicates.",
        len(valid_df),
        len(quarantine_df),
        duplicates_count,
    )
    return valid_df, quarantine_df, audit_summary


def run() -> Dict[str, Any]:
    """Run batch ingestion pipeline on default raw file."""
    print("[Phase 1 Ingestion] Running Big Data Ingestion Pipeline...")
    if not config.RAW_FILE.exists():
        from src.generate_sample_data import generate
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        generate().to_csv(config.RAW_FILE, index=False)

    raw_df = pd.read_csv(config.RAW_FILE)
    valid_df, quaran_df, audit = ingest_batch(raw_df, source_name="data/raw/bank_churn.csv", save_raw=True)
    print(f"  Total Ingested:    {audit['total_received']}")
    print(f"  Valid Records:     {audit['valid_count']}")
    print(f"  Quarantined Rows:  {audit['quarantined_count']}")
    print(f"  Duplicates Curbed: {audit['duplicates_removed']}")
    print(f"  Audit Report:      {AUDIT_REPORT_FILE}")
    return audit


if __name__ == "__main__":
    run()

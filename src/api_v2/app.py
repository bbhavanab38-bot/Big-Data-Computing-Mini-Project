"""FastAPI v2 service for Banking Retention Intelligence Platform."""
import io
import logging
import os
from typing import Optional
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api_v2.schemas import (
    BatchPredictionResponse,
    CustomerInput,
    ModelMetadataResponse,
    PredictionResponse,
    IngestResponse,
    HDFSStatusResponse,
)
from src.api_v2.service import get_model_bundle, predict_batch_dataframe, predict_single
from src.ingestion import ingest_batch
from src.hdfs_utils import check_hdfs_health

logger = logging.getLogger("banking_retention_api")

app = FastAPI(
    title="Banking Retention Intelligence API",
    version="2.0.0",
    description="Modern, calibrated customer churn prediction and risk assessment API.",
)

# Configurable restricted CORS origins
raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501,http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DEFAULT_MAX_BATCH_ROWS = int(os.environ.get("MAX_BATCH_ROWS", "5000"))
DEFAULT_MAX_BATCH_BYTES = int(os.environ.get("MAX_BATCH_BYTES", str(5 * 1024 * 1024)))  # 5 MB


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return structured JSON error details for validation failures (probe protection)."""
    errors = []
    for err in exc.errors():
        field_loc = " -> ".join(str(loc) for loc in err.get("loc", []))
        errors.append({
            "field": field_loc,
            "message": err.get("msg", "Invalid input value"),
            "type": err.get("type", "value_error"),
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "validation_error",
            "message": "Input validation failed. Please check field requirements and bounds.",
            "details": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Prevent internal stack traces from leaking to client in API responses."""
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"status": "error", "message": exc.detail},
            headers=getattr(exc, "headers", None),
        )
    logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "internal_server_error",
            "message": "An unexpected server error occurred. Please contact the administrator.",
        },
    )


def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """Optional API key authentication; enforced only if API_KEY is set in environment."""
    configured_key = os.environ.get("API_KEY") or os.environ.get("RETENTION_API_KEY")
    if configured_key:
        if not x_api_key or x_api_key != configured_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key header.",
            )
    return True


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "banking-retention-api-v2",
        "version": "2.0.0",
    }


@app.get("/v1/model", response_model=ModelMetadataResponse, tags=["Model"], dependencies=[Depends(verify_api_key)])
def get_model_metadata():
    """Return metadata of the active registered production model."""
    bundle = get_model_bundle()
    return {
        "name": bundle.get("name", "Unknown"),
        "model_version": bundle.get("model_version", "unknown"),
        "trained_at": bundle.get("trained_at"),
        "sklearn_version": bundle.get("sklearn_version"),
        "calibration_method": bundle.get("calibration_method"),
        "metrics": bundle.get("metrics", {}),
    }


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
def predict(customer: CustomerInput):
    """Predict churn probability and risk category for a single customer."""
    result = predict_single(customer.model_dump())
    return result


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"], dependencies=[Depends(verify_api_key)])
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction via CSV file upload with strict row and byte limits."""
    # Check filename extension
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV files (.csv) are accepted.",
        )

    # Read content with byte limit enforcement
    content = await file.read()
    if len(content) > DEFAULT_MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed limit of {DEFAULT_MAX_BATCH_BYTES} bytes.",
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to parse CSV file: {str(e)}",
        )

    # Enforce row limit
    if len(df) > DEFAULT_MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size {len(df)} exceeds maximum allowed limit of {DEFAULT_MAX_BATCH_ROWS} rows.",
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file contains 0 data rows.",
        )

    result = predict_batch_dataframe(df)
    return result


@app.get("/v1/hdfs/status", response_model=HDFSStatusResponse, tags=["Big Data Storage"], dependencies=[Depends(verify_api_key)])
def get_hdfs_status():
    """Return live HDFS storage cluster connectivity, NameNode/DataNode status, and capacity."""
    status_info = check_hdfs_health()
    return status_info


@app.post("/v1/ingest", response_model=IngestResponse, tags=["Big Data Ingestion"], dependencies=[Depends(verify_api_key)])
async def ingest_dataset(file: UploadFile = File(...)):
    """Ingest a raw batch CSV file with schema validation, domain checks, deduplication, and quarantine."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Only CSV files (.csv) are accepted for ingestion.",
        )

    content = await file.read()
    if len(content) > DEFAULT_MAX_BATCH_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds maximum allowed limit of {DEFAULT_MAX_BATCH_BYTES} bytes.",
        )

    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to parse CSV file: {str(e)}",
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="CSV file contains 0 data rows.",
        )

    valid_df, quaran_df, audit = ingest_batch(df, source_name=file.filename, save_raw=False)
    return {
        "status": "success",
        "source": file.filename,
        "total_received": audit["total_received"],
        "valid_count": audit["valid_count"],
        "quarantined_count": audit["quarantined_count"],
        "duplicates_removed": audit["duplicates_removed"],
        "quarantine_file": audit["quarantine_file"],
    }



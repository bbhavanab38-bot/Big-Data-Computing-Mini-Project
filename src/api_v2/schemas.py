"""Pydantic v2 schemas with rigorous domain bounds and validation."""
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    Age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    CreditScore: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    Balance: float = Field(..., ge=0.0, description="Current account balance (>= 0)")
    Geography: Literal["France", "Spain", "Germany"] = Field(..., description="Operating country")
    Gender: Literal["Male", "Female"] = Field(..., description="Customer gender")

    Tenure: Optional[int] = Field(default=None, ge=0, le=20, description="Account tenure in years")
    NumOfProducts: Optional[int] = Field(default=None, ge=1, le=4, description="Number of active bank products (1-4)")
    HasCrCard: Optional[int] = Field(default=None, ge=0, le=1, description="Holds credit card (0 or 1)")
    IsActiveMember: Optional[int] = Field(default=None, ge=0, le=1, description="Active membership flag (0 or 1)")
    EstimatedSalary: Optional[float] = Field(default=None, ge=0.0, description="Estimated annual salary (>= 0)")
    TransactionCount: Optional[int] = Field(default=None, ge=0, description="Number of transactions (>= 0)")
    OnlineBankingUsage: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Online banking usage % (0-100)")
    CustomerComplaints: Optional[int] = Field(default=None, ge=0, description="Historical customer complaints (>= 0)")
    LoanAmount: Optional[float] = Field(default=None, ge=0.0, description="Active loan amount (>= 0)")
    LoanStatus: Optional[str] = Field(default=None, description="Current loan status")


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Actionable prediction label")
    churn_probability: float = Field(..., description="Calibrated churn probability [0.0 - 1.0]")
    churn_probability_pct: str = Field(..., description="Human-readable percentage string")
    risk_category: str = Field(..., description="Calibrated risk band (Low, Medium, High)")
    model_version: str = Field(..., description="Registered model version identifier")


class BatchCustomerPrediction(BaseModel):
    CustomerId: Optional[int] = None
    prediction: str
    churn_probability: float
    risk_category: str


class BatchPredictionResponse(BaseModel):
    total_records: int
    churn_likely_count: int
    model_version: str
    predictions: List[BatchCustomerPrediction]


class ModelMetadataResponse(BaseModel):
    name: str
    model_version: str
    trained_at: Optional[str]
    sklearn_version: Optional[str]
    calibration_method: Optional[str]
    metrics: dict


class IngestResponse(BaseModel):
    status: str
    source: str
    total_received: int
    valid_count: int
    quarantined_count: int
    duplicates_removed: int
    quarantine_file: Optional[str] = None


class HDFSStatusResponse(BaseModel):
    available: bool
    status: str
    message: str
    namenode_live: bool
    datanode_live: bool
    capacity_gb: float
    used_mb: float
    mode: str



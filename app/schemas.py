"""
Pydantic schemas for request and response validation.
Defines the exact structure of API input and output.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CustomerFeatures(BaseModel):
    """Input schema — customer data for churn prediction."""

    gender: str = Field(example="Male")
    SeniorCitizen: int = Field(example=0)
    Partner: str = Field(example="Yes")
    Dependents: str = Field(example="No")
    tenure: int = Field(example=12)
    PhoneService: str = Field(example="Yes")
    MultipleLines: str = Field(example="No")
    InternetService: str = Field(example="Fiber optic")
    OnlineSecurity: str = Field(example="No")
    OnlineBackup: str = Field(example="Yes")
    DeviceProtection: str = Field(example="No")
    TechSupport: str = Field(example="No")
    StreamingTV: str = Field(example="Yes")
    StreamingMovies: str = Field(example="No")
    Contract: str = Field(example="Month-to-month")
    PaperlessBilling: str = Field(example="Yes")
    PaymentMethod: str = Field(example="Electronic check")
    MonthlyCharges: float = Field(example=70.35)
    TotalCharges: float = Field(example=844.20)


class PredictionResponse(BaseModel):
    """Output schema — prediction result."""

    churn_prediction: int
    churn_probability: float
    result: str
    confidence: str


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    message: str
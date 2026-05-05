"""
FastAPI application for Customer Churn Prediction.
Provides REST API endpoints for real-time predictions.
"""

import io
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from app.schemas import CustomerFeatures, PredictionResponse, HealthResponse
from src.predict import predict_churn

# ── App setup ────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description="""
    ## Customer Churn Prediction API

    This API predicts whether a telecom customer will churn or not
    based on their account information and usage patterns.

    ### Features
    - Real-time churn prediction
    - Batch prediction via CSV upload
    - Probability score with confidence
    - Built with Random Forest classifier (accuracy: 78.75%)

    ### Author
    Rahul Gaddam — Data Scientist
    """,
    version="1.0.0",
)

# ── CORS middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files & UI ─────────────────────────────────────────
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/ui", include_in_schema=False)
def ui():
    return FileResponse("app/static/index.html")

# ── Load model at startup ─────────────────────────────────────
model = None
feature_names = None


@app.on_event("startup")
async def load_model():
    """Load model and feature names when API starts."""
    global model, feature_names
    try:
        model = joblib.load("models/churn_model.pkl")
        feature_names = joblib.load("models/feature_names.pkl")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"⚠️  Model not found: {e}")
        print("Please train the model first by running: python run_training.py")


# ── Routes ───────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    """Root endpoint — check if API is running."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "message": "Customer Churn Prediction API is live!"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint for deployment monitoring."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "API is healthy and ready to serve predictions."
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn for a single customer.

    Provide customer account details and get:
    - **churn_prediction**: 0 (stays) or 1 (churns)
    - **churn_probability**: probability score between 0 and 1
    - **result**: human readable result
    - **confidence**: confidence percentage
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    try:
        result = predict_churn(
            model=model,
            feature_names=feature_names,
            input_data=customer.dict()
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from a CSV file.

    Upload a CSV file with customer data and get predictions
    for all customers at once. Returns a downloadable CSV.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        original_df = df.copy()

        df = pd.get_dummies(df)
        df = df.reindex(columns=feature_names, fill_value=0)

        predictions = model.predict(df)
        probabilities = model.predict_proba(df)[:, 1]

        original_df["churn_prediction"] = predictions
        original_df["churn_probability"] = probabilities.round(3)
        original_df["result"] = [
            "Will Churn" if p == 1 else "Will Not Churn"
            for p in predictions
        ]
        original_df["confidence"] = [
            f"{round(p * 100, 1)}%" for p in probabilities
        ]

        output = io.StringIO()
        original_df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=churn_predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )
"""
Prediction module for Customer Churn Prediction.
Loads saved model and makes predictions.
"""

import joblib
import pandas as pd
import numpy as np


def load_model():
    """Load trained model and feature names from models/ folder."""
    model = joblib.load("models/churn_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    print("Model loaded successfully!")
    return model, feature_names


def predict_churn(model, feature_names: list, input_data: dict) -> dict:
    """
    Make churn prediction for a single customer.
    
    Args:
        model: Trained RandomForest model
        feature_names: List of feature names used during training
        input_data: Dictionary of customer features
        
    Returns:
        Dictionary with prediction and probability
    """
    # Convert input to dataframe
    df = pd.DataFrame([input_data])

    # One-hot encode
    df = pd.get_dummies(df)

    # Align columns with training data
    df = df.reindex(columns=feature_names, fill_value=0)

    # Make prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 3),
        "result": "Will Churn" if prediction == 1 else "Will Not Churn",
        "confidence": f"{round(float(probability) * 100, 1)}%"
    }
"""
Model training module for Customer Churn Prediction.
Trains Random Forest model and saves artifacts.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def train_model(X_train, y_train) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Model training complete!")
    return model


def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluate model and print all metrics.
    Returns dict of key metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy  : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50)

    return {
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
    }


def save_artifacts(model, feature_names: list):
    """Save trained model and feature names to models/ folder."""
    joblib.dump(model, "models/churn_model.pkl")
    joblib.dump(feature_names, "models/feature_names.pkl")
    print("Model saved to models/churn_model.pkl")
    print("Feature names saved to models/feature_names.pkl")
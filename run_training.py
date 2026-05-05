"""
Main training script for Customer Churn Prediction.
Run this file to train and save the model.

Usage:
    python run_training.py
"""

import pandas as pd
import urllib.request
import os

from src.preprocess import load_data, clean_data, encode_data, split_data
from src.train import train_model, evaluate_model, save_artifacts


def main():
    print("=" * 50)
    print(" CUSTOMER CHURN PREDICTION — MODEL TRAINING")
    print("=" * 50)

    # ── Step 1: Download dataset ──────────────────────────
    data_path = "data/churn.csv"

    if not os.path.exists(data_path):
        print("\n📥 Downloading dataset...")
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        urllib.request.urlretrieve(url, data_path)
        print("Dataset downloaded to data/churn.csv ✅")
    else:
        print("\n Dataset already exists, skipping download ✅")

    # ── Step 2: Load & clean ──────────────────────────────
    print("\n📊 Loading and cleaning data...")
    df = load_data(data_path)
    df = clean_data(df)

    # ── Step 3: Encode ────────────────────────────────────
    print("\n🔠 Encoding data...")
    df = encode_data(df)

    # ── Step 4: Split ─────────────────────────────────────
    print("\n✂️  Splitting data...")
    X_train, X_test, y_train, y_test, feature_names = split_data(df)

    # ── Step 5: Train ─────────────────────────────────────
    print("\n🤖 Training model...")
    model = train_model(X_train, y_train)

    # ── Step 6: Evaluate ──────────────────────────────────
    print("\n📈 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    # ── Step 7: Save ──────────────────────────────────────
    print("\n💾 Saving model artifacts...")
    save_artifacts(model, feature_names)

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Accuracy : {metrics['accuracy'] * 100:.2f}%")
    print(f"ROC-AUC  : {metrics['roc_auc']}")
    print("Model saved to models/churn_model.pkl")
    print("=" * 50)
    print("\nNext step: run the API with:")
    print("uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
    
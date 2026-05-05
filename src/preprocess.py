"""
Data preprocessing module for Customer Churn Prediction.
Handles all data cleaning, encoding, and splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data from filepath."""
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw dataframe:
    - Fix TotalCharges column
    - Drop customerID
    - Remove nulls
    """
    # Fix TotalCharges — it comes as string with spaces
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop customerID — not useful for prediction
    df.drop(columns=["customerID"], inplace=True)

    # Drop rows with nulls
    df.dropna(inplace=True)

    print(f"Data after cleaning: {df.shape[0]} rows")
    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns:
    - Target column Churn → 0/1
    - All other categoricals → one-hot encoding
    """
    # Encode target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode all remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    print(f"Data after encoding: {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame):
    """
    Split into train and test sets.
    Returns X_train, X_test, y_train, y_test and feature names.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, list(X.columns)
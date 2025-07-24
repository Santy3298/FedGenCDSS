# File: preprocessing/data_preprocessor.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_ehr_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the EHR DataFrame:
    - Encode categorical features
    - Scale numeric features
    - Drop identifiers

    Args:
        df (pd.DataFrame): Raw EHR data.

    Returns:
        pd.DataFrame: Cleaned and transformed data ready for model input.
    """
    df = df.copy()

    # Drop Patient_ID
    if "Patient_ID" in df.columns:
        df.drop("Patient_ID", axis=1, inplace=True)

    # Encode Gender
    le = LabelEncoder()
    if "Gender" in df.columns:
        df["Gender"] = le.fit_transform(df["Gender"])

    # Map Diagnosis to class labels
    label_map = {"Healthy": 0, "Prehypertension": 1, "Cardiovascular": 2}
    if "Diagnosis" in df.columns:
        df["Diagnosis"] = df["Diagnosis"].map(label_map)

    # Feature columns and target
    X_cols = [col for col in df.columns if col != "Diagnosis"]
    X = df[X_cols]
    y = df["Diagnosis"] if "Diagnosis" in df.columns else None

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    if y is not None:
        X_scaled["Diagnosis"] = y.values

    return X_scaled


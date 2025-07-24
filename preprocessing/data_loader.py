# File: preprocessing/data_loader.py

import pandas as pd

def load_ehr_data(file_path: str) -> pd.DataFrame:
    """
    Load EHR data from a CSV file.

    Args:
        file_path (str): Path to the EHR CSV file.

    Returns:
        pd.DataFrame: Loaded EHR DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"[INFO] Loaded {data.shape[0]} records from {file_path}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load data from {file_path}: {e}")
        return pd.DataFrame()


# File: utils/config.py

class Config:
    """
    Central configuration for FedGenAI Clinical Decision Support System.
    Modify this file to tune training parameters or global settings.
    """

    # Federated learning parameters
    NUM_CLIENTS = 5
    ROUNDS = 10
    LOCAL_EPOCHS = 5

    # Data paths
    DATA_FILE = "data/clinical_ehr.csv"

    # Model parameters
    INPUT_DIM = 128
    NUM_CLASSES = 3

    # Logging
    LOGGING_LEVEL = "INFO"


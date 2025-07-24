# File: training/client_node_simulation.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model.fedgenai_model import FedGenAIModel

def train_local_model(data, labels, epochs=5):
    """
    Simulates training on a federated client.

    Args:
        data (ndarray): Local training features.
        labels (ndarray): Local training labels.
        epochs (int): Number of local epochs.

    Returns:
        tuple: Trained model weights, accuracy.
    """
    model = FedGenAIModel(input_dim=data.shape[1], num_classes=3)
    model.train(data, labels, epochs=epochs)

    predictions = model.predict(data)
    acc = accuracy_score(labels, predictions)
    return model.get_weights(), acc


# File: training/train_federated.py

import numpy as np
from preprocessing.data_loader import load_ehr_data
from preprocessing.data_preprocessor import preprocess_ehr_data
from training.aggregation import federated_averaging
from training.client_node_simulation import train_local_model

def simulate_federated_training(ehr_file, num_clients=5, rounds=10):
    """
    Simulate full federated training cycle.

    Args:
        ehr_file (str): Path to the CSV data file.
        num_clients (int): Number of simulated clients.
        rounds (int): Total number of federated training rounds.

    Returns:
        list: Final global model weights after training.
    """
    df = load_ehr_data(ehr_file)
    df_clean = preprocess_ehr_data(df)

    X = df_clean.drop("Diagnosis", axis=1).values
    y = df_clean["Diagnosis"].values

    # Split data across clients
    client_data = np.array_split(X, num_clients)
    client_labels = np.array_split(y, num_clients)

    global_weights = None

    for round_idx in range(rounds):
        print(f"\n[ROUND {round_idx+1}] Starting federated round...")
        local_weights = []

        for i in range(num_clients):
            weights, acc = train_local_model(client_data[i], client_labels[i])
            print(f"[Client {i+1}] Local Accuracy: {acc:.4f}")
            local_weights.append(weights)

        global_weights = federated_averaging(global_weights, local_weights)
        print(f"[ROUND {round_idx+1}] Aggregated Global Model")

    return global_weights


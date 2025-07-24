# File: training/aggregation.py

import numpy as np

def federated_averaging(global_weights, local_weights_list):
    """
    Perform Federated Averaging of local model weights from clients.

    Args:
        global_weights (list): List of global model weights.
        local_weights_list (list of list): List of weight lists from each client.

    Returns:
        list: Updated global model weights after averaging.
    """
    new_weights = []
    for weights in zip(*local_weights_list):
        new_weights.append(np.mean(weights, axis=0))
    return new_weights

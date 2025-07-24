from evaluation.metrics import classification_metrics
import numpy as np

def evaluate_model(y_true, y_pred):
    metrics = classification_metrics(y_true, y_pred)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

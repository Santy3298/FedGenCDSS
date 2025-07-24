# fedgenai_model.py

"""
Core model file orchestrating multimodal transformer training under federated setup
with integrated differential privacy for secure and explainable AI.
"""

from model.transformer_module import TransformerModule
from model.privacy_module import DifferentialPrivacyWrapper

class FedGenAIModel:
    """
    FedGenAIModel encapsulates a transformer-based model wrapped with differential privacy
    to enable secure and compliant training across federated environments.
    """

    def __init__(self, config):
        """
        Initializes the FedGenAIModel with transformer and privacy wrapper.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.model = TransformerModule(config)
        self.privacy_layer = DifferentialPrivacyWrapper(self.model, config)

    def forward(self, x):
        """
        Performs a forward pass through the model with privacy safeguards.

        Args:
            x (Tensor): Input tensor (e.g., clinical text, EHR, imaging embeddings).

        Returns:
            Tensor: Output logits or probabilities.
        """
        return self.privacy_layer.forward(x)

    def train(self, dataloader):
        """
        Trains the model on a local dataset using privacy-preserving techniques.

        Args:
            dataloader (DataLoader): A federated or local dataloader object.

        Returns:
            dict: Training metrics (e.g., loss, accuracy).
        """
        return self.privacy_layer.train(dataloader)

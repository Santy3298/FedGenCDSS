# privacy_module.py

"""
Privacy Module: Differential Privacy Wrapper for Secure Federated Training.
Uses the Opacus-style engine to enable private training in compliance with 
privacy-preserving machine learning practices.
"""

from opacus import PrivacyEngine

class DifferentialPrivacyWrapper:
    """
    This class wraps a model with differential privacy using Opacus,
    allowing privacy-preserving training in federated environments.
    """

    def __init__(self, model, config):
        """
        Initializes the privacy engine and attaches it to the model and optimizer.

        Args:
            model (nn.Module): The neural network model.
            config (dict): Dictionary containing training and DP parameters.
        """
        self.model = model
        self.optimizer = config["optimizer"]
        self.privacy_engine = PrivacyEngine()

        # Convert model and optimizer to a private version
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=None,  # Will be attached during training
            noise_multiplier=config["noise_multiplier"],
            max_grad_norm=config["max_grad_norm"]
        )

    def forward(self, x):
        """
        Performs a forward pass through the privatized model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Model output.
        """
        return self.model(x)

    def train(self, dataloader):
        """
        Trains the model using differentially private gradients.

        Args:
            dataloader (DataLoader): Local training data.

        Returns:
            nn.Module: Trained model.
        """
        self.privacy_engine.attach(dataloader)
        self.model.train()

        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self.model(**batch)  # Forward pass with unpacked inputs
            loss.backward()            # Backpropagation
            self.optimizer.step()      # DP step update

        return self.model

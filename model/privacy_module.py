"""
Differential privacy layer for federated settings using Opacus.
"""
from opacus import PrivacyEngine

class DifferentialPrivacyWrapper:
    def __init__(self, model, config):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"])
        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=None,  # Assigned during training
            noise_multiplier=config["noise_multiplier"],
            max_grad_norm=config["max_grad_norm"]
        )

    def forward(self, x):
        return self.model(x)

    def train(self, dataloader):
        self.privacy_engine.attach(dataloader)
        self.model.train()
        for batch in dataloader:
            self.optimizer.zero_grad()
            loss = self.model(**batch)
            loss.backward()
            self.optimizer.step()
        return self.model 

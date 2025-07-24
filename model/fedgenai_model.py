"""
Core model file orchestrating multimodal transformer training under federated setup.
"""
from model.transformer_module import TransformerModule
from model.privacy_module import DifferentialPrivacyWrapper

class FedGenAIModel:
    def __init__(self, config):
        self.model = TransformerModule(config)
        self.privacy_layer = DifferentialPrivacyWrapper(self.model, config)

    def forward(self, x):
        return self.privacy_layer.forward(x)

    def train(self, dataloader):
        return self.privacy_layer.train(dataloader)

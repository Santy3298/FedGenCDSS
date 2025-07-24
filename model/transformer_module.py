"""
Transformer model adapted for clinical multimodal data (ECG, EHR, radiology notes).
"""
import torch
import torch.nn as nn
from transformers import BertModel

class TransformerModule(nn.Module):
    def __init__(self, config):
        super(TransformerModule, self).__init__()
        self.ecg_encoder = nn.LSTM(config["ecg_input"], 64, batch_first=True)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(64 + 768, config["output_dim"])

    def forward(self, ecg, text_input):
        _, (h_ecg, _) = self.ecg_encoder(ecg)
        h_text = self.text_encoder(**text_input).pooler_output
        combined = torch.cat((h_ecg[-1], h_text), dim=1)
        return self.fc(combined)

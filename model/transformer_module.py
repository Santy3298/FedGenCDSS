# transformer_module.py

"""
Transformer Module: Adapted for Clinical Multimodal Data (ECG, EHR, Radiology Notes)

This module combines sequential biosignals (e.g., ECG) with unstructured text (e.g., EHR, notes)
using a hybrid architecture integrating LSTM and BERT, suitable for downstream diagnostic tasks.
"""

import torch
import torch.nn as nn
from transformers import BertModel

class TransformerModule(nn.Module):
    def __init__(self, config):
        """
        Initializes encoders for ECG and clinical text, followed by fusion and output layer.

        Args:
            config (dict): Configuration dictionary with model parameters such as:
                           - "ecg_input": number of ECG input features
                           - "output_dim": number of output classes or regression target size
        """
        super(TransformerModule, self).__init__()
        
        # LSTM for time-series biosignal (e.g., ECG)
        self.ecg_encoder = nn.LSTM(
            input_size=config["ecg_input"],
            hidden_size=64,
            batch_first=True
        )
        
        # BERT encoder for clinical free-text (e.g., EHR, discharge notes)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        
        # Fully connected fusion head
        self.fc = nn.Linear(64 + 768, config["output_dim"])  # 64 from LSTM, 768 from BERT

    def forward(self, ecg, text_input):
        """
        Performs a forward pass over ECG + clinical text.

        Args:
            ecg (Tensor): [batch_size, time_steps, ecg_features]
            text_input (dict): BERT-compatible input dict (input_ids, attention_mask, etc.)

        Returns:
            Tensor: Combined prediction output.
        """
        _, (h_ecg, _) = self.ecg_encoder(ecg)  # h_ecg: [1, batch_size, 64]
        h_ecg = h_ecg[-1]                      # [batch_size, 64]

        h_text = self.text_encoder(**text_input).pooler_output  # [batch_size, 768]
        
        # Concatenate ECG and BERT embeddings
        combined = torch.cat((h_ecg, h_text), dim=1)  # [batch_size, 832]
        
        return self.fc(combined)

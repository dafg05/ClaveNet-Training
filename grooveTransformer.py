import torch
from torch import nn, Tensor
import math
import numpy as np

class GrooveTransformerModel(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, pitches = 9, time_steps = 32):
        """
        TODO
        - Add docstring
        - Add feedworward size hyperparameter
        """
        super(GrooveTransformerModel, self).__init__()

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")

        # hvo dimensions
        self.pitches = pitches
        self.time_steps = time_steps
        
        # layers
        self.linear1 = nn.Linear(3 * self.pitches, d_model)
        self.relu = nn.ReLU()
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.linear2 = nn.Linear(d_model, 3 * self.pitches)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, src):
        x = self.linear1(src)
        x = self.relu(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = self.linear2(output)
        chunks = torch.chunk(output, 3, dim=2)

        hits, velocites, offsets = chunks

        hits = self.sig(hits)
        hits = torch.where(hits > 0.5, 1.0, 0.0)
        velocites = self.sig(velocites)
        offsets = self.tanh(offsets)

        hvo = torch.cat((hits, velocites, offsets), dim=2)
        return hvo

class PositionalEncoding(nn.Module):
    """
    Copy pasted from https://github.com/behzadhaki/BaseGrooveTransformers/blob/main/models/utils.py#L5
    """
    def __init__(self, d_model, max_len=32, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)  # Shape (max_len)
        position = position.unsqueeze(1)  # Shape (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Shape (d_model/2)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        # Insert a new dimension for batch size
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0)]
        return x

        
if __name__ == "__main__":

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    model = GrooveTransformerModel().to(device)
    x = torch.rand(1, 32, 27, device=device)
    hvo = model(x)
    
    testPassed = x.shape == hvo.shape

    print(f"Test passed? {testPassed}")
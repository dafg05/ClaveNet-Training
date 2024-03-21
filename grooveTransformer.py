import torch
from torch import nn, Tensor
import math
import numpy as np
    
class GrooveTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=4, num_layers=6, dim_feedforward=128, dropout=0.0, pitches = 9, time_steps = 32, hit_sigmoid_in_forward = False):
        super(GrooveTransformer, self).__init__()

        self.hit_sigmoid_in_forward = hit_sigmoid_in_forward
        
        # layers
        self.input = InputLayer(embedding_size=3*pitches, d_model=d_model, dropout=dropout, max_len=time_steps)
        self.encoder = Encoder(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, num_encoder_layers=num_layers)
        self.output = OutputLayer(embedding_size=3 * pitches, d_model=d_model, hit_sigmoid_in_forward=hit_sigmoid_in_forward)

    def forward(self, src):
        x = self.input(src)
        encoded = self.encoder(x)
        h, v, o = self.output(encoded)
        return h, v, o
    
    def inference(self, src):
        self.eval()
        with torch.no_grad():
            h, v, o = self.forward(src)
            if not self.hit_sigmoid_in_forward: # if we didn't apply sigmoid in forward, we do it here
                h = torch.sigmoid(h)
            h = torch.where(h > 0.5, 1.0, 0.0)
            return h, v, o

class InputLayer(torch.nn.Module):
    """
    Copy_pasted from https://github.com/behzadhaki/BaseGrooveTransformers/blob/main/models/io_layers.py
    """
    def __init__(self, embedding_size, d_model, dropout, max_len):
        super(InputLayer, self).__init__()

        self.linearIn = torch.nn.Linear(embedding_size, d_model, bias=True)
        self.relu = torch.nn.ReLU()
        self.posEncoding = PositionalEncoding(d_model, max_len, dropout)

    def init_weights(self, initrange=0.1):
        self.linearIn.bias.data.zero_()
        self.linearIn.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        x = self.linearIn(src)
        x = self.relu(x)
        out = self.posEncoding(x)

        return out
    
class Encoder(torch.nn.Module):
    """
    Copy pasted from https://github.com/behzadhaki/BaseGrooveTransformers/blob/main/models/encoder.py
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_encoder_layers):
        super(Encoder, self).__init__()
        # TODO: why do we need to normalize the encoder output?
        norm_encoder = torch.nn.LayerNorm(d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.Encoder = torch.nn.TransformerEncoder(encoder_layer, num_encoder_layers, norm_encoder)

    def forward(self, src):
        # TODO: why are we permuting?
        src = src.permute(1, 0, 2)  # 32xNxd_model
        out = self.Encoder(src)  # 32xNxd_model
        out = out.permute(1, 0, 2)  # Nx32xd_model
        return out

class OutputLayer(torch.nn.Module):
    """
    Based on https://github.com/behzadhaki/BaseGrooveTransformers/blob/main/models/io_layers.py
    """
    def __init__(self, embedding_size, d_model, hit_sigmoid_in_forward=False):
        super(OutputLayer, self).__init__()

        self.embedding_size = embedding_size
        self.linearOut = torch.nn.Linear(d_model, embedding_size)

        self.hit_sigmoid_in_forward = hit_sigmoid_in_forward

    def init_weights(self, initrange=0.1):
        # TODO: why are we initializing the weights like this?
        self.linearOut.bias.data.zero_()
        self.linearOut.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.linearOut(src)
        # TODO: test chunking
        chunks = torch.chunk(src, 3, dim=2)
        hits, velocites, offsets = chunks

        if self.hit_sigmoid_in_forward:
            hits = torch.sigmoid(hits)
        velocites = torch.sigmoid(velocites)
        # TODO: why are we multiplying tanh(offsets) by 0.5?
        offsets = torch.tanh(offsets) * 0.5
        return hits, velocites, offsets

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
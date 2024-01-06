import torch
from torch import nn, Tensor
import math

DEBUG = False

class GrooveTransformerModel(nn.Module):
    def __init__(self, d_model=8, nhead=4, num_layers=3, pitches = 9, time_steps = 32):
        """
        TODO
        - Add docstring
        - Add feedworward size hyperparameter
        """
        super(GrooveTransformerModel, self).__init__()

        # hvo dimensions
        self.pitches = pitches
        self.time_steps = time_steps
        
        # layers
        self.linear1 = nn.Linear(3 * self.pitches, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.linear2 = nn.Linear(d_model, 3 * self.pitches)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, src):
        src = self.linear1(src)
        if DEBUG:
            print(f"src.shape post linear1: {src.shape}")
        src = self.pos_encoder(src)
        if DEBUG:
            print(f"src.shape post pos_encoder: {src.shape}")
        output = self.transformer_encoder(src)
        if DEBUG:
            print(f"output.shape post transformer_encoder: {output.shape}")
        output = self.linear2(output)
        if DEBUG:
            print(f"output.shape post linear2: {output.shape}")
        chunks = torch.chunk(output, 3, dim=2)
        hits, velocites, offsets = chunks
        if DEBUG:
            print("split into hits, velocities and offsets:")
            print(f"hits.shape: {hits.shape}")
            print(f"velocities.shape {velocites.shape}")
            print(f"offsets.shape: {offsets.shape}")
        hits = self.sig(hits)
        hits = torch.where(hits > 0.5, 1.0, 0.0)
        velocites = self.sig(velocites)
        offsets = self.tanh(offsets)

        if DEBUG:
            print("post activation functions:")
            print(f"hits: {hits}")
            print(f"velocities: {velocites}")
            print(f"offsets: {offsets}")
        hvo = torch.cat((hits, velocites, offsets), dim=2)
        return hvo

class PositionalEncoding(nn.Module):
    """
    From pytorch tutorial: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

        
if __name__ == "__main__":

    # test that the shape of the output is correct: aka that it matches the input
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

import torch
from sympy import true
from torch import nn
from einops import rearrange, einsum

from cs336_basics.Transformer_Modules.Linear_Module import Linear


class PositionwiseFeedforwardNetwork(nn.Module):
    def __init__(self, d_model : int, d_ff:int):
        super().__init__()
        self.Linear1 = Linear(d_model,d_ff)
        self.Linear2 = Linear(d_ff,d_model)
        self.Linear3 = Linear(d_model,d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_1 = self.Linear1.forward(x)
        x_2 = self.Linear3.forward(x)
        x_SiLU = x_1 * torch.sigmoid(x_1)
        y = self.Linear2.forward(x_SiLU * x_2)
        return y
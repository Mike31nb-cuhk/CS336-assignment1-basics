# defines RMS Normalization module

import torch
from torch import nn

from tests.conftest import d_model


class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model))
        self.d_model = d_model
        self.eps = eps


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x * x
        RMS = torch.sqrt((torch.sum(x_squared,dim=-1,keepdim=True)/self.d_model + self.eps))
        result = x / RMS * self.g
        return result.to(in_dtype)
# defines linear module

import torch
from torch import nn
from einops import rearrange, einsum

class Linear(nn.Module):

    def __init__(self, in_features : int, out_features : int, device: torch.device | None = None, dtype: torch.dtype | None = None):

        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
        std = sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.weight,0,std,-3*std,3*std)


    def foward(self, x: torch.Tensor) -> torch.Tensor:
        y = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return y
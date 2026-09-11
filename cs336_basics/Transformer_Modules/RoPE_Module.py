import torch
from sympy import true
from torch import nn
from einops import rearrange, einsum


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        k = torch.arange(d_k/2)
        i = torch.arange(max_seq_len)
        k = 2*k/ d_k
        angle_table_at_i_is_1 = 1 / (theta ** k)
        self.theta_table = einsum(angle_table_at_i_is_1, i, "rot_pairs, max_seq_len -> max_seq_len rot_pairs")
        self.sin_table = torch.sin(self.theta_table)
        self.cos_table = torch.cos(self.theta_table)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        a = x[..., 0::2]
        b = x[..., 1::2]
        sin = self.sin_table[token_positions]
        cos = self.cos_table[token_positions]
        # a_times_sin = einsum(a, sin, "... seq_len rot_pairs, ... seq_len rot_pairs -> ... seq_len rot_pairs")
        # b_times_cos = einsum(b, cos, "... seq_len rot_pairs, ... seq_len rot_pairs -> ... seq_len rot_pairs")
        result_a = a * cos - b * sin
        result_b = a * sin + b * cos
        stacked = torch.stack((result_a, result_b), -1)
        return rearrange(stacked," ... rot_pairs two -> ... (rot_pairs two) ")
from pickletools import decimalnl_long

import torch
from torch import nn
from einops import rearrange, einsum

from cs336_basics.Transformer_Modules.Linear_Module import Linear
from cs336_basics.Transformer_Modules.RoPE_Module import RoPE
from cs336_basics.Transformer_Modules.Scaled_Dot_Product_Attention import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model : int, num_heads : int, apply_rope : bool, theta = None, max_seq_len = None):
        super().__init__()
        self.apply_rope = apply_rope
        self.num_heads = num_heads
        self.d_k = d_model/num_heads
        self.Linear_Q = Linear(d_model, d_model)
        self.Linear_K = Linear(d_model, d_model)
        self.Linear_V = Linear(d_model, d_model)
        self.Linear_O = Linear(d_model, d_model)
        if apply_rope:
            self.RoPE = RoPE(theta,self.d_k,max_seq_len)

    def forward(self, in_features : torch.Tensor, token_positions = None):
        seq_len = in_features.shape[-2]
        Q = self.Linear_Q.forward(in_features)
        K = self.Linear_K.forward(in_features)
        V = self.Linear_V.forward(in_features)



        Q = rearrange(Q,"... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", num_heads = self.num_heads)
        K = rearrange(K,"... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", num_heads = self.num_heads)
        V = rearrange(V,"... seq_length (num_heads d_k) -> ... num_heads seq_length d_k", num_heads = self.num_heads)

        if self.apply_rope:
            Q = self.RoPE.forward(Q,token_positions)
            K = self.RoPE.forward(K,token_positions)


        # Masking
        mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool))
        A = scaled_dot_product_attention(Q, K, V, mask)
        A = rearrange(A,"... num_heads seq_length d_k -> ... seq_length (num_heads d_k)")
        return self.Linear_O.forward(A)
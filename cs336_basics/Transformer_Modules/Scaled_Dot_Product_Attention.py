from math import sqrt
import torch
from einops import rearrange, einsum
from cs336_basics.Transformer_Modules.Softmax import softmax

def scaled_dot_product_attention(Q, K, V, mask = None):
    A_pre_softmax = einsum(Q,K,"... n d_k, ... m d_k -> ... n m") / sqrt(Q.shape[-1])
    additive_mask = torch.where(mask,0,float("-inf"))
    A_pre_softmax = torch.add(A_pre_softmax, additive_mask)
    A = softmax(A_pre_softmax, -1)
    return einsum(A,V,"... n m, ... m d_v -> ... n d_v")
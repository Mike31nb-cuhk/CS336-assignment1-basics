from typing import Any

import torch
from torch import nn
from einops import rearrange, einsum

def softmax(input_tensor : torch.Tensor, dimension : int):
    max_entry = torch.max(input_tensor,dim = dimension,keepdim=True).values
    # print("input entry shape!! :")
    # print(input_tensor.shape)
    # print("max entry shape!! :")
    # print(max_entry.shape)
    input_tensor = torch.sub(input_tensor, max_entry)
    logits = torch.exp(input_tensor)
    normalize_factor = torch.sum(logits,dim=dimension, keepdim=True)
    # print(torch.divide(logits, normalize_factor))
    return torch.divide(logits, normalize_factor)
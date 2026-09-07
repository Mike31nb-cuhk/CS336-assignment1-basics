# defines embedding module

import torch
from torch import nn
from einops import rearrange, einsum

class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.embedding_weight, 0, 1, -3, 3)
        self.num_embeddings = num_embeddings

    def foward(self,token_ids: torch.Tensor) -> torch.Tensor:
        y = self.embedding_weight[token_ids]
        return y
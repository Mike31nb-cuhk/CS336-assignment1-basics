import torch
from torch import nn
from einops import rearrange, einsum

from cs336_basics.Transformer_Modules.MultiHead_Self_Attention import MultiheadSelfAttention
from cs336_basics.Transformer_Modules.Positionwise_Feedforward_Network_Module import PositionwiseFeedforwardNetwork
from cs336_basics.Transformer_Modules.RMS_Norm_Module import RMSNorm


class TransformerBlockModule(nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ff : int, theta, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.MultiheadSelfAttention = MultiheadSelfAttention(d_model,num_heads,True, theta, max_seq_len)
        self.PositionwiseFeedforwardNetwork = PositionwiseFeedforwardNetwork(d_model,d_ff)
        self.RMSNorm_MHSA = RMSNorm(d_model)
        self.RMSNorm_PWFFN = RMSNorm(d_model)

    def forward(self, x : torch.Tensor):

        # init MHSA
        token_positions = torch.arange(x.shape[-2])
        token_positions.expand(x.shape[:-1])

        # first sub-layer
        x_RMSNorm = self.RMSNorm_MHSA.forward(x)
        x_MHSA = self.MultiheadSelfAttention.forward(x_RMSNorm,token_positions)
        y = x + x_MHSA

        # second sub-layer
        y_RMSNorm = self.RMSNorm_PWFFN.forward(y)
        y_PWFFN = self.PositionwiseFeedforwardNetwork.forward(y_RMSNorm)
        z = y + y_PWFFN

        return z
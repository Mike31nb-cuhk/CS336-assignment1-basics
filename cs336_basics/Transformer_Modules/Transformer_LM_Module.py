import torch
from torch import nn

from cs336_basics.Transformer_Modules import Embedding_Module, Transformer_Block_Module, RMS_Norm_Module, Linear_Module


class TransformerLM(nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ff : int, theta, context_length, vocab_size : int, num_layers : int):
        super().__init__()
        self.Embedding = Embedding_Module.Embedding(vocab_size,d_model)
        self.TransformerBlocks = nn.ModuleList(Transformer_Block_Module.TransformerBlockModule(d_model, num_heads, d_ff, theta, context_length) for i in range(num_layers))
        self.Final_RMSNorm = RMS_Norm_Module.RMSNorm(d_model)
        self.LM_Head = Linear_Module.Linear(d_model,vocab_size)

    def forward(self,in_indices):

        # Embed indices into vector
        input_tensor = self.Embedding.forward(in_indices)

        # Transformer Blocks!
        for TransformerBlock in self.TransformerBlocks:
            input_tensor = TransformerBlock.forward(input_tensor)

        # Final Norm and LM head
        input_tensor = self.Final_RMSNorm.forward(input_tensor)
        input_tensor = self.LM_Head.forward(input_tensor)
        return input_tensor
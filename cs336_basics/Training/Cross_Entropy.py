import torch
from torch import nn
from einops import rearrange, einsum

def cross_entropy(inputs : torch.Tensor, targets : torch.Tensor):
    inputs = rearrange(inputs,"... vocab_size -> (...) vocab_size")
    targets = rearrange(targets,"... -> (...)")
    max_entry = torch.max(inputs,dim = -1, keepdim=True).values
    inputs = torch.sub(inputs, max_entry)
    exp_inputs = torch.exp(inputs)
    indices = torch.arange(inputs.shape[0])
    logit_ans = inputs[indices,targets]
    exp_logit_summation = torch.sum(exp_inputs,dim=-1)
    loss = -logit_ans + torch.log(exp_logit_summation)
    loss = torch.sum(loss) / loss.shape[0]
    return loss
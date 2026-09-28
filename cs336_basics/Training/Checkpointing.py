import torch
from sympy import false


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    checkpoint = {
    "optimizer" : optimizer.state_dict(),
    "iteration" : iteration,
    "full_model": model}
    torch.save(checkpoint, out)

def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["full_model"].state_dict())
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def load_optimizer(src, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

def load_full_model(src):
    checkpoint = torch.load(src,weights_only=false)
    return checkpoint["full_model"]
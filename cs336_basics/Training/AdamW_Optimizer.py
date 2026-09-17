from collections.abc import Callable, Iterable
from math import sqrt
from typing import Optional
import torch



class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay = 0.99,  betas = (0.9,0.999), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "beta_one":betas[0],"beta_two":betas[1],"epsilon":eps,"lamda":weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta_one = group["beta_one"]
            beta_two = group["beta_two"]
            epsilon = group["epsilon"]
            lamda = group["lamda"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                m = state.get("m", torch.zeros_like(p))  # Get iteration number from the state, or 0.
                v = state.get("v", torch.zeros_like(p))  # Get iteration number from the state, or 0.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data *= 1 - (lr * lamda)
                m = beta_one * m + (1-beta_one)*grad
                v = beta_two * v + (1-beta_two)*grad*grad
                lr_t = lr * sqrt(1-beta_two**t) / (1-beta_one**t)
                p.data -= lr_t * m/(torch.sqrt(v)+epsilon) # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
                state["m"] = m  # Increment iteration number.
                state["v"] = v  # Increment iteration number.
        return loss

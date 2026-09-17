from math import sqrt

import torch

def gradient_clipping(param, l2_norm_max, epsilon = 10e-6):
    l2_norm_squared = 0

    for p in param:
        if p.grad is None:
            continue
        l2_norm_squared += torch.sum(p.grad*p.grad)
    l2_norm = sqrt(l2_norm_squared)

    if l2_norm >= l2_norm_max:
        for p in param:
            if p.grad is None:
                continue
            p.grad *= (l2_norm_max/(l2_norm + epsilon))

    return param
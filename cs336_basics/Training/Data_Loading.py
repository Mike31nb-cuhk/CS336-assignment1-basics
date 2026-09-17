import numpy as np
import torch


def data_loading(x,b,n,device):
    batch_seeds = np.random.randint(0, len(x)-n, b)
    corpus = torch.zeros([b,n],dtype=torch.int32)
    answer = torch.zeros([b,n],dtype=torch.int32)
    for i,seed in enumerate(batch_seeds):
        corpus[i] = torch.from_numpy(x[seed:seed+n])
        answer[i] = torch.from_numpy(x[seed+1:seed+n+1])
    return corpus,answer
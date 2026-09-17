import argparse
import pickle

import numpy as np
import torch

from cs336_basics.Tokenizer import Playground
from cs336_basics.Training import Data_Loading, Cross_Entropy, AdamW_Optimizer, Checkpointing
from cs336_basics.Transformer_Modules import Transformer_LM_Module


def one_time_tokenize():

    indices = Playground.tokenizer_tiny_story_iter()
    np.save("data/indices.npy",np.array(indices))

def training_together(args,device):
    # load params
    b = args.b
    n = args.n
    h = args.h
    d = args.d
    d_ff = args.d_ff
    theta = args.theta
    v = args.v
    n_layers = args.n_layers
    weight_decay = args.weight_decay
    k = args.k
    total_passes = args.total_passes

    # load (b,n) indices
    indices = np.load("data/indices.npy","r")
    in_indices = Data_Loading.data_loading(indices,b,n,device)[0]
    targets = Data_Loading.data_loading(indices,b,n,device)[1]
    print(targets.shape)

    # instantiate transformer and optimizer
    transformer = Transformer_LM_Module.TransformerLM(d,h,d_ff,theta,n,v,n_layers)
    transformer = transformer.to(device)
    optimizer = AdamW_Optimizer.Adam(transformer.parameters(),1e-3,weight_decay)

    # forward pass for k times
    for i in range(total_passes):
        for j in range(k):
            output = transformer.forward(in_indices)
            cross_entropy = Cross_Entropy.cross_entropy(output,targets)
            print(cross_entropy.item())
            cross_entropy.backward()
            optimizer.step()
        Checkpointing.save_checkpoint(transformer,optimizer,(i+1)*k,f"checkpoints/checkpoint_{i}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--v", type=float, default=10000)
    parser.add_argument("--b", type=float, default=50)
    parser.add_argument("--n", type=float, default=256)
    parser.add_argument("--d", type=float, default=512)
    parser.add_argument("--d_ff", type=float, default=1344)
    parser.add_argument("--h", type=float, default=16)
    parser.add_argument("--n_layers", type=float, default=4)
    parser.add_argument("--theta", type=float, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.99)
    parser.add_argument("--k", type=float, default=5)
    parser.add_argument("--total_passes", type=float, default=100)
    device = "cuda:0"
    args = parser.parse_args()
    training_together(args,device)
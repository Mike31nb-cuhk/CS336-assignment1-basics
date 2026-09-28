from operator import index

import torch
import pickle
import random

from sympy import true

from cs336_basics.Training import Checkpointing
from cs336_basics.Tokenizer import bpe_tokenizer
from cs336_basics.Transformer_Modules import Softmax

with open("data/merges.pkl", "rb") as f:
    merges = pickle.load(f)
with open("data/vocabulary.pkl", "rb") as f:
    vocabulary = pickle.load(f)

def decoding(query : str,max_ans_length,temperature : int,top_p):
    tokenizer = bpe_tokenizer.tokenizer(vocabulary, merges, ["<|endoftext|>"])
    query_indicies = torch.Tensor(tokenizer.encode(query)).to(dtype=torch.int).to("cuda:0")
    transformer = Checkpointing.load_full_model("checkpoints/checkpoint_2000.pt")
    transformer = transformer.to("cuda:0")
    transformer.eval()

    for i in range(max_ans_length):

        with torch.inference_mode():
            output = transformer.forward(query_indicies)
        softmax = Softmax.softmax(output,temperature=temperature)
        prob_distribution_sorted, prob_distribution_sorted_indices = torch.sort(softmax,dim = -1, descending=True)
        prob_distribution_cumsum = torch.cumsum(prob_distribution_sorted,-1)

        # top p
        top_p_index = torch.searchsorted(prob_distribution_cumsum[-1],top_p).to("cuda:0")
        top_p_prob_distribution_sorted = prob_distribution_sorted[-1][:1+top_p_index]
        top_p_normalized_prob_distribution_sorted = top_p_prob_distribution_sorted / torch.sum(top_p_prob_distribution_sorted)
        top_p_prob_distribution_sorted_indices = prob_distribution_sorted_indices[-1][:1+top_p_index]
        # for i in range(5):
        #     print(f"token_predict{i}: " + '"'+tokenizer.decode([top_p_prob_distribution_sorted_indices[i].tolist()]) + '",' + "probability: " + str(top_p_normalized_prob_distribution_sorted[i].item()))
        # print()
        # print()
        prob_distribution_cumsum = torch.cumsum(top_p_normalized_prob_distribution_sorted,-1)
        num = random.random()
        index = torch.searchsorted(prob_distribution_cumsum,num).to("cuda:0")
        index = top_p_prob_distribution_sorted_indices[index]
        query_indicies = torch.cat((query_indicies,torch.tensor([index]).to("cuda:0")))

    return tokenizer.decode(query_indicies.tolist())

query = "Once upon a time,"
print(decoding(query, 100, 1, 0.9))

print()
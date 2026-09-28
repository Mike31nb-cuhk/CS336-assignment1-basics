import argparse
import pickle
import math
from pathlib import Path
import wandb
import numpy as np
import torch

from cs336_basics.Tokenizer import Playground
from cs336_basics.Training import Data_Loading, Cross_Entropy, AdamW_Optimizer, Checkpointing, Learning_Rate_Schedule, Gradient_Clipping
from cs336_basics.Transformer_Modules import Transformer_LM_Module


def one_time_tokenize():

    indices = Playground.tokenizer_tiny_story_iter(False)
    np.save("data/train_indices.npy",np.array(indices))
    indices = Playground.tokenizer_tiny_story_iter(True)
    np.save("data/valid_indices.npy",np.array(indices))

# fix this if want to resume run
def resume_training(args, device, wandb_run):
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
    total_passes = math.ceil(args.token_passed/b/n)
    lr = args.lr

    # load checkpoint
    transformer = Checkpointing.load_full_model("checkpoints/lr_sweep/lr_10^0/checkpoint_23002.pt")
    transformer = transformer.to(device)
    # load Adam
    optimizer = AdamW_Optimizer.Adam(transformer.parameters(),lr,weight_decay)
    Checkpointing.load_optimizer(optimizer)
    cross_entropy_cum = 0
    valid_cross_entropy_cum = 0

    # forward pass for k times
    for i in range(23002,total_passes+1):

        # load (b,n) indices
        indices = np.load("data/train_indices.npy", "r")
        in_indices, targets = Data_Loading.data_loading(indices, b, n, device)
        output = transformer.forward(in_indices)
        cross_entropy = Cross_Entropy.cross_entropy(output,targets)
        cross_entropy_cum+=cross_entropy.item()
        cross_entropy.backward()
        optimizer.step()
        optimizer.zero_grad()

        # averaging train_loss every k round
        if i%k == 0:
            train_loss = cross_entropy_cum/k
            cross_entropy_cum = 0
            if i%(10*k) ==0:
                print(f"step{i} train loss: " + str(train_loss))
            wandb_run.log({"train loss" : train_loss}, step=i)

        # collecting validation_loss every k*100 round
        if (i+1)%(k*100) == 0:
            indices = np.load("data/valid_indices.npy", "r")
            for j in range(20):
                in_indices, targets = Data_Loading.data_loading(indices, b, n, device)
                with torch.inference_mode():
                    output = transformer.forward(in_indices)
                valid_cross_entropy_cum += Cross_Entropy.cross_entropy(output, targets)
            valid_loss = valid_cross_entropy_cum / 20
            valid_cross_entropy_cum = 0
            print(f"step{i+1} valid loss: " + str(valid_loss))
            wandb_run.log({"valid loss" : valid_loss}, step=i+1)

        # checkpointing
        if (i+1) % 5000 == 0:
            Checkpointing.save_checkpoint(transformer,optimizer,i+1,f"checkpoints/lr_sweep/lr_{lr:.2f}/checkpoint_{i+1}.pt")

    # final checkpoint
    Checkpointing.save_checkpoint(transformer, optimizer, i+1, f"checkpoints/lr_sweep/lr_{lr:.2f}/checkpoint_{i}_final.pt")

def training_together(args, device, wandb_run):
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
    total_passes = math.ceil(args.token_passed/b/n)
    lr = args.lr

    # instantiate transformer and optimizer
    transformer = Transformer_LM_Module.TransformerLM(d,h,d_ff,theta,n,v,n_layers)
    transformer = transformer.to(device)
    optimizer = AdamW_Optimizer.Adam(transformer.parameters(),lr,weight_decay)
    cross_entropy_cum = 0
    valid_cross_entropy_cum = 0



    # forward pass for k times
    for i in range(1,total_passes+1):

        # load (b,n) indices
        indices = np.load("data/train_indices.npy", "r")
        in_indices, targets = Data_Loading.data_loading(indices, b, n, device)
        output = transformer.forward(in_indices)
        cross_entropy = Cross_Entropy.cross_entropy(output, targets)
        cross_entropy_cum += cross_entropy.item()
        cross_entropy.backward()

        # gradient clipping and lr schedule
        for group in optimizer.param_groups:
            group["lr"] = Learning_Rate_Schedule.learning_rate_schedule(i,lr,lr/10,500,total_passes)
            Gradient_Clipping.gradient_clipping(group["params"],1)

        optimizer.step()
        optimizer.zero_grad()

        # averaging train_loss every k round
        if i % k == 0:
            train_loss = cross_entropy_cum / k
            cross_entropy_cum = 0
            if i % (10 * k) == 0:
                print(f"step{i} train loss: " + str(train_loss))
            wandb_run.log({"train loss": train_loss}, step=i)

        # collecting validation_loss every k*100 round
        if (i + 1) % (k * 100) == 0:
            indices = np.load("data/valid_indices.npy", "r")
            for j in range(20):
                in_indices, targets = Data_Loading.data_loading(indices, b, n, device)
                with torch.inference_mode():
                    output = transformer.forward(in_indices)
                valid_cross_entropy_cum += Cross_Entropy.cross_entropy(output, targets)
            valid_loss = valid_cross_entropy_cum / 20
            valid_cross_entropy_cum = 0
            print(f"step{i + 1} valid loss: " + str(valid_loss))
            wandb_run.log({"valid loss": valid_loss}, step=i + 1)

        # checkpointing
        if (i + 1) % 10000 == 0:
            folder = Path(f"checkpoints/lr_sweep/lr_{lr:.2f}")
            folder.mkdir(parents=True, exist_ok=True)
            Checkpointing.save_checkpoint(transformer, optimizer, i + 1,f"checkpoints/lr_sweep/lr_{lr:.2f}/checkpoint_{i + 1}.pt")

        # final checkpoint
    Checkpointing.save_checkpoint(transformer, optimizer, i + 1,f"checkpoints/lr_sweep/lr_{lr:.2f}/checkpoint_{i+1}_final.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--v", type=int, default=10000)
    parser.add_argument("--b", type=int, default=50)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--d", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--h", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--theta", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--token_passed", type=int, default=327680000)
    device = "cuda:0"
    args = parser.parse_args()

    '''lr sweeping'''

    for i in range(2,11,2):

        args.lr = i*10**-4

        # Start a new wandb run to track this script.
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="cs336mike",
            # Set the wandb project where this run will be logged.
            project="lr-scheduled-sweep-10^-5",
            # Track hyperparameters and run metadata.
            config=args,
            # name
            name=f"lr_sweep_lr={i}*10^-5"
        )

        training_together(args,device,run)

        run.finish()


def resume_run():

    '''resume run'''

    args.lr = 10 ** 0

    # Load the old wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="cs336mike",
        # Set the wandb project where this run will be logged.
        project="my-awesome-project",
        # Track hyperparameters and run metadata.
        id="4fgaru61",
        # resume
        resume="must"
    )
    run.config.update(
        {"lr": args.lr},
        allow_val_change=True,
    )
    resume_training(args, device, run)
    run.finish()
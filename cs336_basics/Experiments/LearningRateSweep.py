import argparse
import wandb
from cs336_basics.Training import Train_Model

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
    parser.add_argument("--weight_decay", type=float, default=0.99)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--token_passed", type=int, default=327680000)
    device = "cuda:0"
    args = parser.parse_args()

    for i in range(10):
        args.lr = 10**i
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="cs336mike",
            # Set the wandb project where this run will be logged.
            project="my-awesome-project",
            # Track hyperparameters and run metadata.
            config=parser.parse_args()
        )
        Train_Model.training_together(args,device,run)
        run.finish()
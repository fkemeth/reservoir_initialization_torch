"""Run Brusselator example."""
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from hyperopt import hp, fmin, tpe

from brusselator.config import config
from brusselator.datasets import BrusselatorParallelDataset
from model import ESN, ESNModel, progress

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

dataset_train = BrusselatorParallelDataset(
    config["DATA"]["n_train"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"]
)
dataset_val = BrusselatorParallelDataset(
    config["DATA"]["n_val"], config["DATA"]["l_trajectories"], config["DATA"]["parameters"]
)
dataset_test = BrusselatorParallelDataset(
    config["DATA"]["n_test"], config["DATA"]["l_trajectories_test"], config["DATA"]["parameters"]
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_train.tt[:-1], dataset_train.input_data[0], label="u")
ax.plot(dataset_train.tt[:-1], dataset_train.v_data[0], label="v")
ax.set_xlabel("t")
plt.legend()
plt.savefig(config["PATH"] + "data.pdf")
plt.close()


# Create PyTorch dataloaders for train and validation data
dataloader_train = DataLoader(
    dataset_train,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
dataloader_val = DataLoader(
    dataset_val,
    batch_size=config["TRAINING"]["batch_size"],
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

def objective(args):
    reservoir_size, ridge_factor = args
    network = ESN(
        config["MODEL"]["input_size"],
        config["MODEL"]["reservoir_size"],
        config["MODEL"]["hidden_size"],
        config["MODEL"]["input_size"],
        config["MODEL"]["scale_rec"],
        config["MODEL"]["scale_in"],
        config["MODEL"]["leaking_rate"],
    )

    model = ESNModel(
        dataloader_train,
        dataloader_val,
        network,
        learning_rate=config["TRAINING"]["learning_rate"],
        offset=config["TRAINING"]["offset"],
        ridge_factor=config["TRAINING"]["ridge_factor"],
        device=config["TRAINING"]["device"],
    )

    loss = model.train(ridge=config["TRAINING"]["ridge"])
    return loss

space = [hp.choice('reservoir_size', [128, 512, 2048, 4096]),
         hp.choise('ridge_factor', [1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5])]

best = fmin(objective, space, algo=tpe.suggest, max_evals=2)

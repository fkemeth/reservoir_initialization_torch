"""Run Brusselator example."""
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from datasets import BrusselatorDataset
from model import ESN, ESNModel, progress
from config import config

torch.set_default_dtype(config["TRAINING"]["dtype"])

if not os.path.exists(config["PATH"]):
    os.makedirs(config["PATH"])

dataset_train = BrusselatorDataset(config["DATA"]["n_train"],
                                   config["DATA"]["l_trajectories"],
                                   config["DATA"]["parameters"])
dataset_val = BrusselatorDataset(config["DATA"]["n_val"],
                                 config["DATA"]["l_trajectories"],
                                 config["DATA"]["parameters"])
dataset_test = BrusselatorDataset(config["DATA"]["n_test"],
                                  config["DATA"]["l_trajectories_test"],
                                  config["DATA"]["parameters"])
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

network = ESN(
    config["MODEL"]["input_size"],
    config["MODEL"]["reservoir_size"],
    config["MODEL"]["input_size"],
    config["MODEL"]["scale_rec"],
    config["MODEL"]["scale_in"],
    config["MODEL"]["leaking_rate"])

model = ESNModel(
    dataloader_train,
    dataloader_val,
    network,
    learning_rate=config["TRAINING"]["learning_rate"],
    offset=config["TRAINING"]["offset"])

model.load_network(config["PATH"] + "model_")

# warmup = config["DATA"]["max_warmup"]
warmup = 300
predictions, _ = model.integrate(
    torch.tensor(
        dataset_test.input_data[0][:warmup], dtype=torch.get_default_dtype()).to(model.device),
    T=dataset_test.input_data[0].shape[0] - warmup - 1,
)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(dataset_test.tt[:-1], dataset_test.input_data[0][:, 0], label="true")
if len(predictions.shape) > 1:
    ax.plot(dataset_test.tt[:-1], predictions[:, 0], label="prediction")
else:
    ax.plot(dataset_test.tt[:-1], predictions, label="prediction")
ax.axvline(x=dataset_test.tt[warmup], color="k")
ax.set_xlabel("$t$")
ax.set_ylabel("$x$")
plt.savefig("fig/predictions.pdf")
plt.show()


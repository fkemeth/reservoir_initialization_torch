import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from utils import config, LorenzDataset, ESN, ESNModel, progress

torch.set_default_dtype(config["TRAINING"]['dtype'])


def main():
    dataset_train = LorenzDataset(config["DATA"]["n_train"],
                                  config["DATA"]["l_trajectories"],
                                  config["MODEL"]["input_size"], verbose=True)
    dataset_val = LorenzDataset(config["DATA"]["n_val"],
                                config["DATA"]["l_trajectories"],
                                config["MODEL"]["input_size"], verbose=False)
    # Create PyTorch dataloaders for train and validation data
    dataloader_train = DataLoader(dataset_train, batch_size=config["TRAINING"]["batch_size"],
                                  shuffle=True, num_workers=8, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=config["TRAINING"]["batch_size"],
                                shuffle=False, num_workers=8, pin_memory=True)

    # batch = next(iter(dataloader_train))

    # Create the network architecture
    network = ESN(config["MODEL"]["input_size"], config["MODEL"]["reservoir_size"],
                  config["MODEL"]["input_size"])
    print(network)

    # Create model wrapper around architecture
    # Contains train and validation functions
    model = ESNModel(dataloader_train, dataloader_val, network,
                     learning_rate=config["TRAINING"]["learning_rate"])

    if config["TRAINING"]["ridge"]:
        config["TRAINING"]["epochs"] = 1
    # Train for the given number of epochs
    progress_bar = tqdm(range(0, config["TRAINING"]["epochs"]),
                        leave=True, position=0, desc=progress(0, 0))
    train_loss_list = []
    val_loss_list = []
    for _ in progress_bar:
        train_loss = model.train(ridge=config["TRAINING"]["ridge"])
        val_loss = model.validate()
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        progress_bar.set_description(progress(train_loss, val_loss))
    model.save_network(config["PATH"]+'model_')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(config["TRAINING"]["epochs"]), train_loss_list, label='train loss')
    ax.plot(np.arange(config["TRAINING"]["epochs"]), val_loss_list, label='validation loss')
    ax.set_xlabel('epoch')
    ax.set_yscale('log')
    plt.legend()
    plt.savefig('fig/loss.pdf')
    plt.show()

    warmup = 200
    predictions, _ = model.integrate(torch.tensor(
        dataset_val.x[0][:warmup], dtype=torch.get_default_dtype()),
        T=dataset_val.x[0].shape[0]-warmup-1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(dataset_val.tt[:-1], dataset_val.x[0][:, 0])
    ax.plot(dataset_val.tt[:-1], predictions[:, 0])
    ax.axvline(x=dataset_val.tt[warmup], color='k')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    plt.savefig('fig/predictions.pdf')
    plt.show()


if __name__ == "__main__":
    main()

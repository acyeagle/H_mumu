import os
import pickle as pkl
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tomllib
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model_generation.focal_loss import BinaryFocalLoss


class Trainer:
    """
    Class for doing the training on a provided model with the provided data.
    See "Train" section of model_generation/config.toml for definition of the input vars.
    Returns a finished model and saves loss plots.
    """

    def __init__(
        self,
        training_data,
        validation_data,
        hyperparams,
        batch_size,
        epochs,
        early_stop=False,
        patience=None,
        early_threshold=0,
        label_smoothing=0,
        device=None,
    ):
        # Training data as a tuple of NumPy arrays (x_data, y_data), weights
        self.training_data = training_data
        # Convert the data to a Torch DataLoader, for optimal training
        self.train_dataloader = self._make_dataloader(training_data, batch_size, device)
        self.valid_dataloader = self._make_dataloader(
            validation_data, batch_size, device
        )

        # Determine from y_data shape if we're training in binary or multiclass mode
        self.binary_classification = len(self.training_data[0][1][0]) == 1
        if self.binary_classification:
            # self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
            #self.loss_fn = torch.nn.BCELoss(reduction="none")
            self.loss_fn = BinaryFocalLoss(gamma=4, reduction="none")
        else:
            # self.loss_fn = torch.nn.NLLLoss(reduction="none")
            self.loss_fn = torch.nn.CrossEntropyLoss(
                reduction="none", label_smoothing=label_smoothing
            )

        # Any config parameters specific for the optimizer
        self.hyperparams = hyperparams

        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        if patience:
            self.patience = patience
        self.threshold = early_threshold

        # Lists to store average epoch losses
        self.training_loss = []
        self.validation_loss = []

    ### Init helpers ###

    def _make_dataloader(self, data, batch_size, device):
        """
        Converts the data (tuple of Numpy arrays) into a DataLoader object
        """
        (x_data, y_data), weights = data
        if device is not None:
            x_data = torch.tensor(x_data, device=device, dtype=torch.double)
            y_data = torch.tensor(y_data, device=device, dtype=torch.double)
            weights = torch.tensor(weights, device=device, dtype=torch.double)
            dataset = TensorDataset(x_data, y_data, weights)
        else:
            dataset = TensorDataset(
                torch.Tensor(x_data), torch.Tensor(y_data), torch.Tensor(weights)
            )
        if batch_size == 0:
            dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    def _set_optimizer(self, model):
        """
        Inits the optimizer.
        """
        algo = self.hyperparams["algo"]
        hypers = {k: v for k, v in self.hyperparams.items() if k != "algo"}
        # Case switch
        if algo == "SGD":
            opt = torch.optim.SGD
        elif algo == "Adam":
            opt = torch.optim.Adam
        else:
            raise ValueError("Optimizer config should specify SGD or Adam as algo.")
        # Actually init and set
        self.optimizer = opt(model.parameters(), **hypers)

    ### Utility functions ###

    def plot_losses(self, valid=True, log=False, show=False):
        """
        Creates a scatter plot showing the average epoch loss
        for both training and validation. Saves to file, unless
        show = True
        """
        plt.clf()
        if valid:
            plt.plot(
                self.validation_loss, color="tab:red", marker="o", label="validation"
            )
            filename = "training_loss_plot_with_valid"
        else:
            filename = "training_loss_plot"
        plt.plot(self.training_loss, color="tab:green", marker="o", label="training")
        if self.early_stop:
            i = np.argmin(self.validation_loss)
            plt.scatter(
                i,
                self.validation_loss[i],
                label="min",
                color="black",
                marker="D",
                zorder=6,
            )
        if log:
            plt.yscale("log")
            filename += "_log"
        plt.xlabel("Training epoch")
        plt.ylabel("Avg. loss")
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig(filename + ".svg", bbox_inches="tight")

    def write_loss_data(self, output_name="epoch+valid_loss"):
        """
        Write the loss data to pickle (in case wanted to investigate after the fact)
        """
        data = (self.training_loss, self.validation_loss)
        with open(output_name + ".pkl", "wb") as f:
            pkl.dump(data, f)

    ### Main training functions ###

    def train_single_epoch(self, model):
        """
        The main training loop. Runs a single epoch
        and runs validation
        """
        epoch_loss = 0
        model.train()
        for sample in self.train_dataloader:
            # Every data instance is an input + label pair
            inputs, labels, weights = sample
            # Zero your gradients for every batch!
            self.optimizer.zero_grad()
            # Make predictions for this batch
            outputs = model(inputs)
            # Compute the loss and its gradients
            loss = self.loss_fn(outputs, labels)
            loss = (loss * weights).mean()
            epoch_loss += loss.item()
            loss.backward()
            # Adjust learning weights
            self.optimizer.step()
        # Add epoch training loss data to validator
        avg_loss = epoch_loss / len(self.train_dataloader)
        self.training_loss.append(avg_loss)
        # Run validation
        valid_loss = self.validate(model)
        self.validation_loss.append(valid_loss)
        return model

    def validate(self, model):
        """
        Runs the model on non-training data to evaluate progress
        """
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for sample in self.valid_dataloader:
                x, y, weight = sample
                guess = model(x)
                loss = self.loss_fn(guess, y)
                loss = (weight * loss).mean()
                total_loss += loss.item()
        avg_loss = total_loss / len(self.valid_dataloader)
        return avg_loss

    ### Training loop definitions ###

    def train_fixed(self, model):
        """
        Runs the training loop for a fixed number of epochs
        """
        for i in tqdm(range(self.epochs), unit="epochs"):
            model = self.train_single_epoch(model)
        return model

    def train_early_stop(self, model):
        """
        Runs the training loop until the validation data performs worse
        """
        best_model = model.state_dict()
        bad_trains = 0
        i = 0
        with tqdm(desc="Training in early stop mode", unit="epochs") as pbar:
            while True:
                model = self.train_single_epoch(model)
                if (
                    self.validation_loss[-1]
                    <= min(self.validation_loss) - self.threshold
                ):
                    best_model = model.state_dict()
                    bad_trains = 0
                else:
                    bad_trains += 1
                if bad_trains > self.patience:
                    break
                i += 1
                pbar.update(1)
        model.load_state_dict(best_model)
        return model

    ### Main (call this function) ###

    def train(self, model):
        """
        Main externally called function.
        Simple switch to dispatch a training run.
        """
        self._set_optimizer(model)
        if self.early_stop and self.patience is not None:
            model = self.train_early_stop(model)
        else:
            model = self.train_fixed(model)
        print("Training finished!")
        return model

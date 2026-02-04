import argparse
import multiprocessing
import os
import tomllib
from pprint import pprint
from uuid import uuid1 as uuid
import shutil
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import TensorDataset

from dataset import JetDataset
from pandas_loader import PandasLoader
from network_2 import Network
from testing import Tester
from training import Trainer
from kfold import KFolder


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="NN_Generator",
        description="For a given dataset and config file, creates a network, trains it, and runs testing",
    )
    parser.add_argument("-c", "--config", required=True, help="the .toml config file")
    parser.add_argument(
        "-d",
        "--datafile",
        required=True,
        help="the .root file to use for testing and training events",
    )
    parser.add_argument(
        "-l",
        "--label",
        required=False,
        help="some string to append to the output folder name",
    )
    args = parser.parse_args()
    return args


def build_layer_list(config):
    # Modify layer_list to have input and output layers
    layer_list = config["network"]["layer_list"]
    # Look at the number of data columns
    input_size = len(config["dataset"]["data_columns"])
    config["network"]["layer_list"] = [input_size] + layer_list + [1]
    return config

def make_dataset(df, for_inference, device, data_columns, **kwargs):
    """
    Converts the dataframe into a DataLoader object
    """
    # Parse input features
    x = df[data_columns].values
    x = torch.tensor(x, device=device, dtype=torch.double)
    if for_inference:
        idx = df.index.values
        idx = torch.tensor(idx)
        dataset = TensorDataset(x, idx)
    else:
        # Parse targets
        y = df.Label.values
        y = y.reshape([len(y), 1])
        y = torch.tensor(y, device=device, dtype=torch.double)
        # Parse training weights
        w = df.Training_Weight.values
        w = w.reshape([len(w), 1])
        w = torch.tensor(w, device=device, dtype=torch.double)
        # Make dataset
        dataset = TensorDataset(x, y, w)
    return dataset

def get_device():
    # Set device for training (cpu or cuda)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Checking CUDA...")
        print(f"\tDevice count: {torch.cuda.device_count()}")
        print(f"\tCurrent device: {torch.cuda.current_device()}")
    else:
        raise ValueError("Killing! Must use CUDA (lxplus CPU training spikes memory)")
    return device


if __name__ == "__main__":
    # Set correct multiprocessing (needed for DataLoader parallelism)
    #multiprocessing.set_start_method("spawn", force=True)
    # Read the CLI arguments
    args = get_arguments()
    device = get_device()

    # Read in config and datasets from args
    print("Reading config...")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    config = build_layer_list(config)

    # Load and split
    print("Reading root file --> dataframe...")
    pl = PandasLoader(args.datafile, **config["dataset"])
    df = pl.load_to_dataframe()
    if 'NN_Output' in df.columns:
        df.rename(columns={'NN_Output' : 'VBFNet_Output'}, inplace=True)
    df = df[df.Signal_Fit & (df.VBFNet_Output >= 0.0)]
    df.reset_index(inplace=True, drop=True)

    # Init the tester
    print("Init'ing tester...")
    tester = Tester(df, **config["testing"] | config["dataset"])

    # Init the output directory
    print("Init'ing output dir...")
    run_name = str(uuid())
    if args.label:
        run_name += f"_{args.label}"
    dir_init = os.getcwd()
    print("\t", run_name)
    os.chdir(config["meta"]["results_dir"])
    os.mkdir(run_name)
    os.chdir(run_name)
    path = os.path.join(dir_init, args.config)
    shutil.copy(path, "./")
    base_dir = os.getcwd()
    print("Working dir:", base_dir)

    # Begin k-fold training loop
    print("Performing k-fold split...")
    kfolder = KFolder(k=config["splitting"]["k"])
    for i, (test_idx, temp_idx) in enumerate(kfolder.split(df)):
        print(temp_idx)
        print(test_idx)
        print("* ON FOLD:", i)
        test_df = df.loc[test_idx].copy()
        # Init an output dir for this fold
        os.mkdir(f"{i}_fold")
        os.chdir(f"{i}_fold")

        # Split validation off of training
        print("Train/valid splitting...")
        temp_df = df.loc[temp_idx]
        size = config["splitting"]["validation_size"]
        train_df, valid_df = train_test_split(
            temp_df, test_size=size, stratify=temp_df["process"]
        )
        train_df = train_df.copy()
        valid_df = valid_df.copy()

        if config['dataset']['renorm_inputs']:
            print("Applying input renorm to train, valid, & test DFs...")
            train_df, (mean, std) = pl.renorm_inputs(train_df, mean=None, std=None)
            valid_df, _ = pl.renorm_inputs(valid_df, mean=mean, std=std)
            test_df, _ = pl.renorm_inputs(test_df, mean=mean, std=std)
            with open('renorm_vars.pkl', 'wb') as f:
                pkl.dump((mean, std), f)
        
        # Parse the pd.DFs to torch.Datasets, init trainer
        print("Converting DFs --> custom datasets...")
        
        train_data = make_dataset(
            train_df, for_inference=False, device=device, **config['dataset']
        )
        valid_data = make_dataset(
            valid_df, for_inference=False, device=device, **config['dataset']
        )
        test_data = make_dataset(
            test_df, for_inference=True, device=device, **config['dataset']
        )

        # Init train
        print("Init'ing trainer...")
        trainer = Trainer(
            train_data,
            valid_data,
            config["optimizer"],
            **config["training"],
        )

        # Do the actual training
        print("Starting the train loop...")
        model = Network(device, **config["network"])
        model = trainer.train(model)

        # Save results
        print("Done training. Saving model...")
        trainer.plot_losses()
        with open("model.torch", "wb") as f:
            torch.save(model, f)

        # Inference this batch
        print("Running inference...")

        tester.test(model, test_data)

        # Back to the top and do it all again
        os.chdir(base_dir)

    # Save the final inference plots
    print("K-fold complete! Saving final plots...")
    tester.testing_df.to_pickle("evaluated_testing_df.pkl")
    tester.make_hist(log=True)
    tester.make_hist(log=False, weight=False, norm=True)
    tester.make_multihist(log=True)
    tester.make_roc_plot()
    tester.make_roc_plot(log=True)
    tester.make_transformed_stackplot()

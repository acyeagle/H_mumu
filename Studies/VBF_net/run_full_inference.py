import argparse
import multiprocessing
import os
import tomllib
from pprint import pprint
from uuid import uuid1 as uuid
import shutil
import pickle as pkl
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import TensorDataset

from pandas_loader import PandasLoader
from network_2 import Network
from testing import Tester
from kfold import KFolder


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="NN_Generator",
        description="For a given dataset and config file, creates a network, trains it, and runs testing",
    )
    parser.add_argument(
        "--directory",
        required=True,
        help="the results directory to infer on. should have a directories for each k fold model",
    )
    parser.add_argument(
        "--datafile",
        required=True,
        help="the input pandas df pkl file to use for inference",
    )
    args = parser.parse_args()
    return args

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

    os.chdir(args.directory)

    # Read in config and datasets from args
    print("Reading config...")
    c = glob("config*.toml")[0]
    with open(c, "rb") as f:
        config = tomllib.load(f)

    # Load and split
    print("Reading in dataframe...")
    pl = PandasLoader(args.datafile, **config["dataset"])
    df = pl.load_to_dataframe()
    if 'NN_Output' in df.columns:
        df.rename(columns={'NN_Output' : 'VBFNet_Output'}, inplace=True)
    df = df[df.Signal_Fit]
    df.reset_index(inplace=True, drop=True)
    df.rename({'NN_Output': 'VBFNet_Output'}, inplace=True)

    # Init the tester
    print("Init'ing tester...")
    tester = Tester(df, **config["testing"] | config["dataset"])

    # Begin k-fold training loop
    print("Performing k-fold split...")
    kfolder = KFolder(k=config["splitting"]["k"], fold_idx_only=True)
    for k, fold_idx in enumerate(kfolder.split(df)):
        print("* ON FOLD:", k)
        print(type(fold_idx))
        print(fold_idx)
        test_df = df.loc[fold_idx].copy()
        os.chdir(f"{k}_fold")

        if config['dataset']['renorm_inputs']:
            print("Applying input renorm to test DF...")
            with open('renorm_vars.pkl', 'rb') as f:
                mean, std = pkl.load(f)
            test_df, _ = pl.renorm_inputs(test_df, mean=mean, std=std)
        
        # Parse the pd.DFs to torch.Datasets
        test_data = make_dataset(
            test_df, for_inference=True, device=device, **config['dataset']
        )

        # Load and run model
        model = torch.load("model.torch", weights_only=False)
        tester.test(model, test_data)

        # Back to the top and do it all again
        os.chdir(args.directory)

    # Save the final inference plots
    print("K-fold complete! Saving final plots...")
    tester.testing_df.to_pickle("FULLEVAL_evaluated_testing_df.pkl")


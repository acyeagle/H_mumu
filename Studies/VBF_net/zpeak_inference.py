import argparse
import multiprocessing
import os
import tomllib
from pprint import pprint
from uuid import uuid1 as uuid
from glob import glob
from pprint import pprint
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset

from dataset import JetDataset
from pandas_loader import PandasLoader
from network_2 import Network
from testing import Tester

def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results_dir",
        required=True,
        help="the directory containing model and config etc to run inference on",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="the directory containing model and config etc to run inference on",
    )
    parser.add_argument(
        "-z",
        "--zpeak_data",
        required=True,
        help="The Zpeak file to inference",
    )
    args = parser.parse_args()
    return args

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

if __name__ == "__main__":
    args = get_arguments()

    device = get_device()

    os.chdir(args.results_dir)

    # Read in config and datasets from args
    print("Reading config...")
    with open('config_nokfold_topvars.toml', "rb") as f:
        config = tomllib.load(f)
    # Load model
    model = torch.load('model.torch', weights_only=False)

    # Load data
    print("Reading root file --> dataframe...")
    pl = PandasLoader(args.zpeak_data, **config["dataset"])
    df = pl.load_to_dataframe()

    # Init the tester
    print("Init'ing tester...")
    tester = Tester(df.copy(), **config["testing"] | config["dataset"])

    # Prep data
    if config['dataset']['renorm_inputs']:
        with open('renorm_var.pkl', 'rb') as f:
            mean, std = pkl.load(f)
        print("Applying input renorm DF...")
        df, (mean, std) = pl.renorm_inputs(df, mean=mean, std=std)
    test_data = make_dataset(
        df, for_inference=True, device=device, **config['dataset']
    )

    # Inference this batch
    print("Running inference...")
    tester.test(model, test_data)

    # Save the final DF
    os.chdir(args.output_dir)
    stem = os.path.basename(args.zpeak_data)
    print("Saving output plots...")
    tester.testing_df.to_pickle(f"EVALUATED_{stem}")
    with open(f'{stem}_used_args.txt', 'w') as f:
        pprint(args, stream=f)
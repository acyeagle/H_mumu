import argparse
import multiprocessing
import os
import tomllib
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from dataset import JetDataset
from rootloader import RootLoader
from network import Network
from testing import Tester


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
        "-r",
        "--rootfile",
        required=True,
        help="the .root file to use for testing and training events",
    )
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="The results directory in question",
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


if __name__ == "__main__":
    # Set correct multiprocessing (needed for DataLoader parallelism)
    multiprocessing.set_start_method("spawn", force=True)
    # Read the CLI arguments
    args = get_arguments()
    device = get_device()

    # Read in config and datasets from args
    print("Reading config...")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Load and split
    print("Reading root file --> dataframe...")
    rl = RootLoader(args.rootfile, **config["dataset"])
    df = rl.load_to_dataframe()
    # Reduce datasize for a quick test
    print("Downsampling DF (for fast test)")
    _, df = train_test_split(df, test_size=0.10, stratify=df['process'])
    df.reset_index(inplace=True)


    # Init the output directory
    os.chdir(args.directory)

    # Init the tester
    print("Init'ing tester...")
    tester = Tester(df, device, **config["testing"] | config["dataset"])
    
    # Load model
    model = torch.load("model.torch", weights_only=False)

    # Inference this batch
    print("Running inference...")
    tester.test(model, None)

    # Save the final inference plots
    print("Saving final plots...")
    tester.testing_df.to_pickle("evaluated_testing_df.pkl")
    tester.make_hist(norm=False, log=True)
    tester.make_hist(norm=True, log=True)
    tester.make_hist(norm=True, log=False)
    tester.make_multihist()
    tester.make_multihist(log=True)
    tester.make_roc_plot()

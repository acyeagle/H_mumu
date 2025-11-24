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
        "-d",
        "--directory",
        required=True,
        help="The results directory in question",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Set correct multiprocessing (needed for DataLoader parallelism)
    multiprocessing.set_start_method("spawn", force=True)
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    print("Reading config...")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Init the output directory
    os.chdir(args.directory)

    # Init the tester
    print("Init'ing tester...")
    df = pd.read_pickle('evaluated_testing_df.pkl')
    tester = Tester(df, device=None, **config["testing"] | config["dataset"])
    
    # Save the final inference plots
    print("Saving final plots...")
    tester.make_hist(norm=False, log=True)
    tester.make_hist(norm=True, log=True)
    tester.make_hist(norm=True, log=False)
    tester.make_multihist()
    tester.make_multihist(log=True)
    tester.make_roc_plot()
    tester.make_roc_plot(log=False)

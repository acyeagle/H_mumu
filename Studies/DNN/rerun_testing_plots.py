import argparse
import os
import pickle as pkl
from datetime import datetime
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
import tomllib
from model_generation.test import Tester
from model_generation.onnx_exporter import export_to_onnx

"""
This is the high-level script describing a full train/test cycle.
Usual workflow is:
    - load config
    - create Dataloader and read in .root samples
    - create Preprocessor and set/transform training weights
    - create Trainer and run training. Plot losses.
    - create Tester and run inference. Produce all plots.
    - save a copy of the model and parameters used. Results location set in config. 
This (since k-fold) repeats the train process for each subflod, 
then puts all the inference results together at the end.
"""


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
        "--resultsdir",
        required=True,
        help="Results directory containing the evaluated testing df",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    # Go to the neede results dir
    os.chdir(args.resultsdir)

    # Make some vars to init tester
    device = None
    testing_df = pd.read_pickle("evaluated_testing_df.pkl")
    dummy_data = (np.zeros(4), np.zeros(4)), np.zeros(4)

    testing_df
    # Init
    tester = Tester(
        dummy_data,
        testing_df,
        device=device,
        **config["testing"] | config["dataloader"],
    )

    # Plot/save
    tester.make_hist(log=False, weight=True, norm=True)
    tester.make_hist(log=True, weight=True, norm=False)
    tester.make_multihist(log=True, weight=True)
    tester.make_stackplot(log=True)
    tester.make_transformed_stackplot()
    tester.make_roc_plot(log=True)
    tester.make_roc_plot(log=False)
    if config['dataloader']['classification'] == "multiclass":
        tester.plot_multiclass_probs()
    tester.make_thist()

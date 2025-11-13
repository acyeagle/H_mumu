import argparse
import os
import pickle as pkl
from datetime import datetime
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
import tomllib
import torch
from model_generation.dataloader import DataLoader
from model_generation.kfold import KFolder
from model_generation.network import Network
from model_generation.preprocess import Preprocessor
from model_generation.test import Tester
from model_generation.train import Trainer
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
        "--rootfile",
        required=True,
        help="the .root file to use for testing and training events",
    )
    parser.add_argument(
        "-d",
        "--results_dir",
        required=False,
        help="Directory containing the models and stuff"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    # Load in all train/valid/test entries as a big DF
    # Sets initial things like class weight, sample_name, label
    dataloader = DataLoader(**config["dataloader"])
    df = dataloader.build_master_df(args.rootfile)
    print("Initial Dataframe size:", len(df))
    df = dataloader._add_labels(df)
    if dataloader.classification == "multiclass":
        df = dataloader._add_multiclass_labels(df)
    df = dataloader._add_class_weights(df)

    # Init our other objects
    preprocessor = Preprocessor(**config["preprocess"] | config["dataloader"])
    kfold = KFolder(**config["kfold"])

    # Set device for training (cpu or cuda)
    if config["meta"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Moving to CUDA...")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("*WARNING: training on CPU (slow)")
        device = None

    print("Postprocess Dataframe size:", len(df))
    # Start the k-fold training loop
    os.chdir(args.results_dir)
    start = datetime.now()
    test_results = None
    for i, (test_df, everything_else) in enumerate(
        kfold.split(df, k=config["kfold"]["k"])
    ):
        train_df, valid_df, empty_df = dataloader._split_dataframe(everything_else)
        assert len(empty_df) == 0

        # Renorm sets to m=0 s=1 separately.
        # Don't want to leak info from test into train
        train_df, (m, s) = dataloader._dispatch_input_renorm(train_df)
        test_df, _ = dataloader._dispatch_input_renorm(test_df, m, s)

        a = len(train_df)
        b = len(valid_df)
        c = len(test_df)
        print("Train DF size:", a)
        print("Valid DF size:", b)
        print("Test DF size", c)
        print("Total:", a + b + c)

        # Parse into (x,y), w tuples (aka "datasets")
        test_data = dataloader.df_to_dataset(test_df)

        # Init testing objects and run
        model = torch.load(f"{i}_fold/trained_model_{i}.torch")

        tester = Tester(
            test_data,
            test_df,
            device=device,
            **config["testing"] | config["dataloader"],
        )

        tester.test(model)

        # Stash the output of the testing inference
        cols = ["NN_Output"]
        if tester.classification == "multiclass":
            cols += [f"Prob_{p}" for p in tester.processes]
        results = tester.testing_df[cols]
        if test_results is None:
            test_results = results
        else:
            test_results = pd.concat([test_results, results])

    # Done!
    end = datetime.now()

    # Combine the inference results with the main dataframe
    print("Size of testing results:", len(test_results))
    print("Size of existing DF:", len(df))
    df = pd.merge(df, test_results['NN_Output'], how='left', left_index=True, right_index=True)
    print("Size of merged DF:", len(df))
    df.to_pickle("evaluated_testing_df.pkl")

    # Plot/save
    tester.testing_df = df
    tester.make_hist(log=False, weight=True, norm=True)
    tester.make_hist(log=True, weight=True, norm=False)
    tester.make_multihist(log=True, weight=True)
    tester.make_stackplot(log=True)
    tester.make_transformed_stackplot()
    tester.make_roc_plot(log=True)
    tester.make_roc_plot(log=False)
    if dataloader.classification == "multiclass":
        tester.plot_multiclass_probs()
    tester.make_thist()

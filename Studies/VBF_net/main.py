import argparse
import multiprocessing
import os
import tomllib
from pprint import pprint
from uuid import uuid1 as uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

from dataset import JetDataset
from rootloader import RootLoader
from network import Network
from testing import Tester
from training import Trainer


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
        "-l",
        "--label",
        required=False,
        help="some string to append to the output folder name",
    )
    parser.add_argument(
        "-d",
        "--down_sample",
        required=False,
        default=0,
        type=float,
        help="Downsample for testing",
    )

    args = parser.parse_args()
    return args


def build_layer_list(config):
    # Modify layer_list to have input and output layers
    layer_list = config["network"]["layer_list"]
    # Look at the number of data columns
    input_size = 5 * config["dataset"]["max_jets"]
    config["network"]["layer_list"] = [input_size] + layer_list + [1]
    return config


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
    config = build_layer_list(config)

    # Load and split
    print("Reading root file --> dataframe...")
    rl = RootLoader(args.rootfile, **config["dataset"])
    df = rl.load_to_dataframe()
    # Reduce datasize for a quick test
    if args.down_sample != 0:
        print(f"Downsampling DF to {args.down_sample} the size (for fast test)")
        df = df.sample(frac=args.down_sample).reset_index()

    # Init the tester
    print("Init'ing tester...")
    tester = Tester(df, device, **config["testing"] | config["dataset"])

    # Init the output directory
    print("Init'ing output dir...")
    run_name = str(uuid())
    if args.label:
        run_name += f"_{args.label}"
    print("\t", run_name)
    os.chdir(config["meta"]["results_dir"])
    os.mkdir(run_name)
    os.chdir(run_name)
    base_dir = os.getcwd()
    print("Working dir:", base_dir)

    # Begin k-fold training loop
    print("Performing k-fold split...")
    skf = StratifiedKFold(n_splits=config["splitting"]["k"])
    for i, (temp_idx, test_idx) in enumerate(skf.split(df, df.process)):
        print("* ON FOLD:", i)
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

        # Parse the pd.DFs to torch.Datasets, init trainer
        print("Converting DF --> custom dataset...")
        train_data = JetDataset(
            train_df, weight_col="Training_Weight", **config["dataset"]
        )
        valid_data = JetDataset(
            valid_df, weight_col="Training_Weight", **config["dataset"]
        )
        trainer = Trainer(
            train_data,
            valid_data,
            device,
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
        tester.test(model, test_idx)

        # Back to the top and do it all again
        os.chdir(base_dir)

    # Save the final inference plots
    print("K-fold complete! Saving final plots...")
    tester.testing_df.to_pickle("evaluated_testing_df.pkl")
    tester.make_hist(log=True)
    tester.make_multihist(log=True)
    tester.make_roc_plot()

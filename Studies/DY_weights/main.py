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
from model_generation.test import Tester
from model_generation.train import Trainer

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
        "-l",
        "--label",
        required=False,
        help="some string to append to the output folder name",
    )
    args = parser.parse_args()
    return args


def write_parameters(start, end, config, dataset, variables_used):
    """
    Writes the parameters used in this model generation run to a text file.
    """
    frmt = "%Y-%m-%d %H:%M:%S"
    with open("used_params.txt", "w") as f:
        f.write(f"Started training: {start.strftime(frmt)}\n")
        f.write(f"Finished training: {end.strftime(frmt)}\n")
        f.write(f"Dataset used: {dataset}\n")
        f.write("\n")
        pprint(config, stream=f)
        f.write("\n")
        f.write("Variables used for network input vector:\n")
        pprint(variables_used, stream=f)


def build_layer_list(config, dataloader):
    # Modify layer_list to have input and output layers
    layer_list = config["network"]["layer_list"]
    input_size = len(dataloader.data_columns)
    output_size = 1
    config["network"]["layer_list"] = [input_size] + layer_list + [output_size]
    return config


if __name__ == "__main__":
    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    # Alter the dataloader config to set test_size to zero
    # Testing size determined by fold
    config["dataloader"]["test_size"] = 0

    # Init the output directory
    run_name = str(uuid())
    if args.label:
        run_name += f"_{args.label}"
    os.chdir(config["meta"]["results_dir"])
    os.mkdir(run_name)
    os.chdir(run_name)
    base_dir = os.getcwd()

    # Load in all train/valid/test entries as a big DF
    # Sets initial things like class weight, sample_name, label
    dataloader = DataLoader(**config["dataloader"])
    df = dataloader.build_master_df(args.rootfile)
    print("Initial Dataframe size:", len(df))
    df = dataloader._add_labels(df)
    df = dataloader._add_class_weights(df)
    df = dataloader._add_training_weights(df)
    if config["dataloader"]['equalize_weights']:
        df = dataloader._equalize_train_weights(df)

    # Add the correct layer-list to the config (instead of just hidden)
    p = pd.unique(df.process)
    config = build_layer_list(config, dataloader)
    # Init our other objects
    kfold = KFolder(**config["kfold"])

    # Set device for training (cpu or cuda)
    if config["meta"]["use_cuda"] and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Moving to CUDA...")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
    else:
        print("*WARNING: training on CPU (slow)")
        raise ValueError("Killing cause a CPU train sucks")

    # Stash the initial DF (so can compare after)
    df.to_pickle("initial_loaded_df.pkl")

    # Start the k-fold training loop
    models = {}
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
        valid_df, _ = dataloader._dispatch_input_renorm(valid_df, m, s)
        test_df, _ = dataloader._dispatch_input_renorm(test_df, m, s)

        a = len(train_df)
        b = len(valid_df)
        c = len(test_df)
        print("Train DF size:", a)
        print("Valid DF size:", b)
        print("Test DF size", c)
        print("Total:", a + b + c)

        # Parse into (x,y), w tuples (aka "datasets")
        train_data = dataloader.df_to_dataset(train_df)
        valid_data = dataloader.df_to_dataset(valid_df)
        test_data = dataloader.df_to_dataset(test_df)

        # Init training run specific objects
        model = Network(device=device, **config["network"])
        trainer = Trainer(
            train_data,
            valid_data,
            config["optimizer"],
            **config["training"],
            device=device,
        )
        tester = Tester(
            test_data,
            test_df,
            device=device,
            **config["testing"] | config["dataloader"],
        )

        # Run the traininig!
        model = trainer.train(model)
        tester.test(model)

        # Stash the output of the testing inference
        results = tester.testing_df
        if test_results is None:
            test_results = results
        else:
            test_results = pd.concat([test_results, results])

        # Prepare an output sub-directory to save files
        run_name = f"{i}_fold"
        os.mkdir(run_name)
        os.chdir(run_name)
        print("Saving outputs to", run_name)

        # Save output files
        trainer.plot_losses()
        trainer.plot_losses(valid=True)
        trainer.write_loss_data()
        outname = f"trained_model_{i}"

        # Save model (torch)
        with open(outname + ".torch", "wb") as f:
            torch.save(model, f)

        # Back to the main kfold dir
        os.chdir(base_dir)

    # Done!
    end = datetime.now()
    write_parameters(start, end, config, args.rootfile, dataloader.data_columns)

    # Combine the inference results with the main dataframe
    print("Size of testing results:", len(test_results))
    print("Size of existing DF:", len(df))
    print("Size of merged DF:", len(df))
    test_results.to_pickle("evaluated_testing_df.pkl")

    # Plot/save
    tester.testing_df = test_results
    tester.make_hist(log=False, weight=True, norm=False)
    tester.make_hist(log=True, weight=True, norm=False)
    tester.make_hist(log=False, weight=True, norm=True)
    tester.make_hist(log=True, weight=True, norm=True)
    tester.make_roc_plot(log=True)
    tester.make_roc_plot(log=False)

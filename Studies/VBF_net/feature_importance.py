
import numpy as np
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from pprint import pprint
import tomllib
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import pickle as pkl


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="Feature importance runner",
        description="Calculates Shapely values for the provided model and data",
    )
    parser.add_argument("-r", "--results_dir", required=True, help="Path to the results directory containing evaluated_testing_df and model")
    args = parser.parse_args()
    return args

def load_model_and_data():
    datapath = "evaluated_testing_df.pkl"
    df = pd.read_pickle(datapath)
    _, df = train_test_split(
        df, test_size=0.05, stratify=df["process"]
    )
    modelpath = "model.torch"
    model = torch.load(modelpath)
    return df, model


def main(model, df, data_cols):
    # Load/prep/clean
    x_sample = df[data_cols].values
    # Define our objective function
    device = torch.device('cuda')
    f = lambda x: model(torch.tensor(x, device=device, dtype=torch.double))
    # Generate the explanatory model
    explainer = shap.Explainer(f, x_sample, feature_names=data_cols)
    shap_values = explainer(x_sample)
    # Do the outputs!
    shap.summary_plot(shap_values=shap_values, feature_names=data_cols)
    plt.savefig('feature_importance.png')
    with open("shapely_values.pkl", "wb") as f:
        pkl.dump(shap_values, f)

if __name__ == '__main__':
    args = get_arguments()
    print("Input args:")
    pprint(args)
    os.chdir(args.results_dir)
    df, model = load_model_and_data()
    with open("config_nokfold.toml", "rb") as f:
        config = tomllib.load(f)
    data_cols = config['dataset']['data_columns']
    main(model, df, data_cols)

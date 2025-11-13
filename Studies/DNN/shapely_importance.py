
import numpy as np
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from pprint import pprint

from model_generation.parse_column_names import parse_column_names

def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="Feature importance runner",
        description="Calculates Shapely values for the provided model and data",
    )
    parser.add_argument("-r", "--results_dir", required=True, help="Path to the results directory containing evaluated_testing_df and model")
    parser.add_argument("-c", "--columns_config", required=True, help="Path to YAML defining column names")
    parser.add_argument("-k", "--k_fold", required=True, help="K-fold value used for this model run")
    args = parser.parse_args()
    return args

def get_column_info(columns_config):
    with open(columns_config, "r") as file:
        config = yaml.safe_load(file)
    config = config["vars_to_save"]
    data_columns = parse_column_names(config, column_type="data")
    return data_columns

def load_model_and_data(results_dir):
    print("Loading from:", results_dir)
    datapath = os.path.join(results_dir, "evaluated_testing_df.pkl")
    df = pd.read_pickle(datapath)
    modelpath = os.path.join(results_dir, "0_fold", "trained_model_0.torch")
    model = torch.load(modelpath)
    return df, model

def prepare_df(df, data_cols, k_fold):
    print("Preparing DF...")
    mask = np.mod(df.FullEventId.values, k_fold) == 0
    x_eval = df[mask]
    x_sample = df[~mask]
    for col in data_cols:
        # Get mean and std from TRAIN
        m = np.mean(x_sample[col])
        s = np.std(x_sample[col])
        # Renorm TRAIN and EVAL
        x_sample[col] = (x_sample[col] - m)/s
        x_eval[col] = (x_eval[col] - m)/s
    return x_eval[data_cols].values, x_sample[data_cols].values

def main(columns_config, results_dir, k_fold):
    # Load/prep/clean
    df, model = load_model_and_data(results_dir)
    data_cols = get_column_info(columns_config)
    x_eval, x_sample = prepare_df(df, data_cols, k_fold)
    # Define our objective function
    if not torch.cuda.is_available():
        raise AttributeError("Can't use CUDA! Aborting to avoid nuking CPU time...")
    device = torch.device("cuda")
    f = lambda x: model(torch.tensor(x, device=device, dtype=torch.double))
    # Generate the explanatory model
    explainer = shap.Explainer(f, x_sample, feature_names=data_cols)
    shap_values = explainer.shap_values(x_eval)
    # Do the outputs!
    os.chdir(results_dir)
    shap.summary_plot(shap_values=shap_values, feature_names=data_cols)
    plt.savefig('feature_importance.png')
    with open("shapely_values.pkl", "wb") as f:
        pkl.dump(shap_values, f)

if __name__ == '__main__':
    args = get_arguments()
    print("Input args:")
    pprint(args)
    main(args.columns_config, args.results_dir, int(args.k_fold))

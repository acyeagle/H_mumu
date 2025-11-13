import warnings
from glob import glob
from math import floor
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import toml
import uproot
import yaml
from model_generation.parse_column_names import parse_column_names

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataLoader:
    """
    Class for doing the inital data reading and processing.
    This sets the Labels and Class_Weights.
    Additional Train_Weight preprocessing is done in Preprocessor.
    This is usually the first workflow step (after reading configs and boilerplate)
    """

    def __init__(
        self,
        columns_config,
        valid_size,
        test_size,
        selection_cut,
        renorm_inputs,
        **kwargs,
    ):
        self._get_column_info(columns_config)
        self.valid_size = valid_size
        self.test_size = test_size
        if selection_cut == "":
            selection_cut = None
        self.selection_cut = selection_cut
        self.renorm_inputs = renorm_inputs

    def _get_column_info(self, columns_config):
        with open(columns_config, "r") as file:
            config = yaml.safe_load(file)
        config = config["vars_to_save"]
        self.data_columns = parse_column_names(config, column_type="data")
        self.header_columns = parse_column_names(config, column_type="header")
        self.all_columns = parse_column_names(config, column_type="all")
        print("Header columns:")
        pprint(self.header_columns)
        print("Data columns for network input:")
        pprint(self.data_columns)

    ### Functions for modifying the loaded dataframe ###
    ### Everything in this section takes a df and returns a df ###

    def _ensure_float(self, df):
        """
        Quick wrapper to make sure everything fed to the net is float (no ints!)
        """
        df[self.data_columns] = df[self.data_columns].astype("float")
        return df

    def _add_labels(self, df):
        """
        Adds the training labels [0, 1] for bkg and sig (resp.)
        """
        df["Label"] = df.weight_MC_Lumi_pu.apply(
            lambda x: 1 if x > 0 else 0
        ).astype(float)
        return df


    def _add_class_weights(self, df):
        """
        Class_Weight is the corrected MC_Lumi_pu applied for all plotting
        """
        df["Class_Weight"] = np.abs(df.weight_MC_Lumi_pu.values.copy())
        return df

    def _add_training_weights(self, df):
        df['Training_Weight'] = df.Class_Weight.values.copy()
        return df

    
    def _equalize_train_weights(self, df):
        total = np.sum(df.Training_Weight)
        weights = df.Training_Weight.values.copy()
        for label in [0, 1]:
            mask = df.Label == label
            subtotal = weights[mask].sum()
            factor = (1 / 2) * (total / subtotal)
            weights[mask] = weights[mask] * factor
        df["Training_Weight"] = weights
        return df
        


    def _apply_gauss_renorm(self, df, means=None, stds=None):
        """
        Takes a data column and maps all values
        to average = 0 and std = 1
        """
        print("Applying gaussian renorm...")
        # Calculate mean and std if not provided
        if means is None or stds is None:
            means = np.zeros(len(self.data_columns))
            stds = np.zeros(len(self.data_columns))
            for i, col in enumerate(self.data_columns):
                data = df[col].values
                m = np.mean(data)
                s = np.std(data)
                means[i] = m
                stds[i] = s
                print(f"{col} - Mean: {m}, StDev: {s}")
        # Apply the renorm
        for i, col in enumerate(self.data_columns):
            data = df[col].values.copy()
            m = means[i]
            s = stds[i]
            df[col] = (data - m) / s
        return df, (means, stds)


    def _dispatch_input_renorm(self, df, m=None, s=None):
        """
        Simple switch for applying input renorms
        """
        # Apply variables renorm
        if self.renorm_inputs == "no":
            return df, (None, None)
        elif self.renorm_inputs == "gauss":
            return self._apply_gauss_renorm(df, m, s)
        else:
            raise ValueError("no or gauss only options for input renorming")

    ### Primary worker functions ###

    def _root_to_dataframe(self, filename):
        """
        Turns a single .root file into a Pandas Df.
        """
        cols = self.all_columns
        with uproot.open(filename) as f:
            tree = f["Events"]
            df = tree.arrays(
                cols, 
                cut=self.selection_cut, 
                library="pd"
            )
        for col in ['process', 'era', 'dataset']:
            df[col] = df[col].astype(str)
        return df

    def _split_dataframe(self, data):
        """
        Turns the given (whole) Df into three Dfs,
        each containing a relative portion of each process' events (i.e., striated).
        valid_size and train_size dictate the fractional size of each new Df.
        test_size gets whatever is left in the Df.
        """
        training_df = pd.DataFrame(columns=data.columns)
        valid_df = pd.DataFrame(columns=data.columns)
        testing_df = pd.DataFrame(columns=data.columns)
        # Add each category to each dataframe
        for category in pd.unique(data.Label):
            selected = (
                data[data.Label == category].sample(frac=1)
            )
            number = len(selected)
            valid_size = floor(number * self.valid_size)
            test_size = floor(number * self.test_size)
            # Add a size number of rows to the df
            valid_df = pd.concat([valid_df, selected[:valid_size]])
            selected = selected[valid_size:]
            testing_df = pd.concat([testing_df, selected[:test_size]])
            selected = selected[test_size:]
            training_df = pd.concat([training_df, selected])
        # Shuffle and return
        training_df = training_df.sample(frac=1)
        valid_df = valid_df.sample(frac=1)
        testing_df = testing_df.sample(frac=1)
        return training_df, valid_df, testing_df

    def build_master_df(self, directory):
        """
        Goes over each input root file and builds one big Pandas Df.
        """
        df = None
        for filename in glob(f"{directory}/*.root"):
            print(f"Generating testing/training samples sets from {filename}")
            if df is None:
                df = self._root_to_dataframe(filename)
            else:
                df = pd.concat([df, self._root_to_dataframe(filename)])
        #df = self._add_sample_names(df)
        df.reset_index(inplace=True, drop=True)
        return df

    ### Main runner function ###

    def generate_dataframes(self, directory):
        """
        The main function that should be called externally.
        Takes a directory containing the input root files
        and returns split Pandas Dfs for training, testing, validation
        """
        # Build a single dataframe from root files
        df = self.build_master_df(directory)
        # Modify the main dataframe
        df = self._add_labels(df)
        df = self._add_class_weights(df)
        # Split and return
        train_df, valid_df, test_df = self._split_dataframe(df)
        return train_df, valid_df, test_df

    def df_to_dataset(self, df):
        """
        Takes a Pandas Df and returns a "dataset"
        A dataset is what Trainer and Validator expect.
        It's a tuple of Numpy arrays of the form:
        (x_data, y_data), weights
        """
        # Get labels
        y = df.Label.values
        y = y.reshape([len(y), 1])
        # Get the input vectors
        x = df[self.data_columns].values
        # Class weights
        w = df.Training_Weight.values
        w = w.reshape([len(w), 1])
        # Return (x,y) tuple
        return (x, y), w

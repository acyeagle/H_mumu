import pandas as pd
import numpy as np


class PandasLoader:

    def __init__(self, filepath, signal_types, equalize_for_training=True, **kwargs):
        self.signal_types = signal_types
        self.filepath = filepath
        self.equalize_for_training = equalize_for_training

    ### Private helper funcs ###

    def _load_file(self, filepath):
        df = pd.read_pickle(filepath)
        for col in ['era', 'process', 'dataset']:
            df[col] = df[col].astype(str)
        return df

    def _add_labels(self, df):
        df["Label"] = df.process.apply(lambda x: 1 if x in self.signal_types else 0)
        return df

    def _set_class_weight(self, df):
        df["Class_Weight"] = df.weight_MC_Lumi_pu.copy()
        return df

    def _set_training_weight(self, df, norm=True):
        train_weights = df.Class_Weight.values.copy()
        train_weights = np.abs(train_weights)
        df["Training_Weight"] = train_weights
        return df

    def _equalize_train_weights(self, df):
        """
        Scale signal to equal background weight
        """
        sig_weight = df[df.Label == 1].Training_Weight.sum()
        bkg_weight = df[df.Label == 0].Training_Weight.sum()
        mask = df.Label == 1
        scale = np.ones(len(df))
        scale[mask] = bkg_weight / sig_weight
        df["Training_Weight"] *= scale
        return df

    ### Main function ###

    def load_to_dataframe(self):
        df = self._load_file(self.filepath)
        df = self._add_labels(df)
        df = self._set_class_weight(df)
        df = self._set_training_weight(df)
        if self.equalize_for_training:
            df = self._equalize_train_weights(df)
        return df

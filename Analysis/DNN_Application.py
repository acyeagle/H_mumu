from __future__ import annotations
import os, sys
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import tomllib
import pickle as pkl
from pprint import pprint

from Studies.DNN.model_generation.parse_column_names import parse_column_names
import FLAF.Common.Utilities as Utilities
import Analysis.H_mumu as analysis
from FLAF.Common.Utilities import DeclareHeader
from Corrections.Corrections import Corrections


class DNNProducer:
    def __init__(self, cfg, payload_name, period):
        print("DNN Producer init!")
        # cfg is H_mumu/configs/global.yaml
        self.cfg = cfg
        self.global_cfg = self._load_global_config()
        self.payload_name = payload_name
        self.period = period
        self._load_framework()
        self.parity, self.input_features = self._load_dnn_config()
        # Columns for tmp file
        self.vars_to_save = self.input_features  # + self.corrected_input_features
        # Final columns
        self.cols_to_save = [
            f"{self.payload_name}_{col}" for col in self.cfg["columns"]
        ]
        self.models = self._load_models()

    ### Init helpers ###

    def _load_framework(self):
        sys.path.append(os.environ["ANALYSIS_PATH"])
        for header in [
            "FLAF/include/Utilities.h",
            "include/Helper.h",
            "include/HmumuCore.h",
            "FLAF/include/AnalysisTools.h",
            "FLAF/include/AnalysisMath.h",
            "FLAF/include/HistHelper.h",
        ]:
            DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)

    def _load_global_config(self):
        filepath = os.path.join(os.environ["ANALYSIS_PATH"], "config", "global.yaml")
        with open(filepath, "r") as f:
            global_config = yaml.safe_load(f)
        return global_config

    def _load_models(self):
        """
        Load in the trained ONNX models
        """
        directory = os.path.join(os.environ["ANALYSIS_PATH"], "Analysis", "models")
        models = []
        renorm_vars = []
        for i in range(self.parity):
            # The model itself
            filename = os.path.join(directory, f"trained_model_{i}.onnx")
            model = ort.InferenceSession(filename)
            models.append(model)
        return models

    def _load_dnn_config(self):
        """
        Read in the config used to train the network and input features used
        """
        directory = os.path.join(os.environ["ANALYSIS_PATH"], "Studies", "DNN")
        filepath = os.path.join(directory, "configs", "config.toml")
        with open(filepath, "rb") as f:
            config = tomllib.load(f)
        parity = config["kfold"]["k"]
        # Input features
        filepath = os.path.join(directory, "configs", "columns_config.yaml")
        with open(filepath, "r") as f:
            columns_config = yaml.safe_load(f)
        input_features = parse_column_names(
            columns_config["vars_to_save"], column_type="data"
        )
        return parity, input_features

    ### Functions for running the inference ###

    def prepare_dfw(self, dfw, dataset_name):
        print("*********** Running prepare_dfw...")
        analysis.InitializeCorrections(self.period, dataset_name, stage="HistTuple")
        corrections = Corrections.getGlobal()
        dfw = analysis.DataFrameBuilderForHistograms(
            dfw.df, self.global_cfg, self.period, corrections
        )
        print(
            f"when creating DataFrameBuilderForHistograms, dfw has {dfw.df.Count().GetValue()} entries"
        )
        dfw = analysis.PrepareDFBuilder(dfw)
        print(f"on top of PrepareDFBuilder it has {dfw.df.Count().GetValue()} entries")
        return dfw

    def ApplyDNN(self, branches):
        print("*********** Running ApplyDNN...")
        nEvents = len(branches)
        event_number = np.array(getattr(branches, "FullEventId"))
        input_array = np.array(
            [getattr(branches, feature_name) for feature_name in self.input_features]
        ).transpose()
        all_predictions = np.zeros([nEvents, self.parity])
        for parityIdx, sess in enumerate(self.models):
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            predictions = sess.run([label_name], {input_name: input_array})[0]
            mask = (event_number % self.parity) != parityIdx
            predictions[mask] = 0
            all_predictions[:, parityIdx] = predictions.reshape(predictions.shape[0])
        final_predictions = np.sum(all_predictions, axis=1)
        # Last save the branches
        branches["NNOutput"] = final_predictions.transpose().astype(np.float32)
        print("Finishing call, memory?")
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {mem_mb:.2f} MB")
        return branches

    def run(self, array):
        print("*********** Running DNN producer")
        array = self.ApplyDNN(array)
        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    del array[col]
        # Rename the branches
        for col in self.cfg["columns"]:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")
        return array

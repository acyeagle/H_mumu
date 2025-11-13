import argparse
import os
import pickle as pkl
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tomllib
from model_generation.dataloader import DataLoader

"""
Quick n' dirty lil script to load all the parsed anaTuples into a Pandas df.
Quick way to do counts/sums for events/weights and check acceptance. 
"""


@dataclass
class ARGS:
    config: str
    rootfile: str


args = ARGS(
    "configs/config.toml",
    "/eos/user/a/ayeagle/H_mumu/root_files/v2vbfselnew/Run3_2022",
)

# Read in config and datasets from args
with open(args.config, "rb") as f:
    config = tomllib.load(f)

# Comment/uncomment to load EVERYTHING
# config["dataloader"]["selection_cut"] = ""

dataloader = DataLoader(**config["dataloader"])
df = dataloader.build_master_df(args.rootfile)
df = dataloader._add_labels(df)
if dataloader.classification == "multiclass":
    df = dataloader._add_multiclass_labels(df)
df = dataloader._add_class_weights(df)

# Stats tab preprocess
cols = ["final_weight", "weight_MC_Lumi_pu", "Class_Weight"]
for col in cols:
    x = df[col].values.copy()
    np.clip(x, a_min=0, a_max=None, out=x)
    name = f"{col}_POS"
    df[name] = x
pos_cols = [x for x in df.columns if "_POS" in x]


def tabulate_stats(value):
    counts = pd.DataFrame(df.value_counts(value))
    sums = df[[value] + cols + pos_cols].groupby(value).sum()
    stats = pd.merge(counts, sums, left_index=True, right_index=True)
    print("*** Grouped By:", value)
    print(stats)
    outname = f"group_by_{value}_stats.csv"
    stats.to_csv(outname)


tabulate_stats("sample_name")
tabulate_stats("source_file")

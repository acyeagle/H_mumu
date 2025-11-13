import argparse
import os
import pickle as pkl
from glob import glob

import numpy as np
import pandas as pd
import ROOT as root


"""
Quick n' dirty lil script to load all the parsed anaTuples into a Pandas df.
Quick way to do counts/sums for events/weights and check acceptance. 
"""


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="InitPop counter",
        description="Count up initial populations in anaTuples",
    )
    parser.add_argument(
        "-d",
        "--anaTuples_dir",
        required=True,
        help="The directorr containing the anaTuples"
    )
    args = parser.parse_args()
    return args


def main(anaTuples_dir):
    os.chdir(args.anaTuples_dir)
    data = pd.DataFrame(columns=["Era", "Sample", "N_anaTuples", "Initial_Population"])
    for era in os.listdir():
        print("On era:", era)
        os.chdir(era)
        for sample in os.listdir():
            pattern = os.path.join(sample, "anaTuple*.root")
            anaTuples_list = glob(pattern)
            total = 0
            try:
                for anaTuple in anaTuples_list:
                    total += get_count(anaTuple)
            except AttributeError:
                total = 0
            row = [era, sample, len(anaTuples_list), total]
            print("\t", row)
            data.loc[len(data)] = row
        os.chdir(anaTuples_dir)
    return data



def get_count(anaTuple_file):
    f = root.TFile(anaTuple_file)
    report = f.Get("Report")
    return report.GetBinContent(1)


if __name__ == "__main__":
    args = get_arguments()
    base_dir = os.getcwd()
    os.chdir(args.anaTuples_dir)
    data = main(args.anaTuples_dir)
    os.chdir(base_dir)
    data.to_csv("initial_population_count.csv")

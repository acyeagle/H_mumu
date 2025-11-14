
import os
from FLAF.Common.Setup import Setup
import argparse
from glob import glob

import ROOT
import yaml
import uproot

import Analysis.H_mumu as analysis
from FLAF.Common.Setup import Setup

ROOT.gSystem.Load("libRIO")
ROOT.gInterpreter.Declare('#include <string>')

def get_args():
    parser = argparse.ArgumentParser(description="Parse AnaTuples to add DNN variables")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="The configuration yaml specifiying the samples and any selection cuts",
    )
    parser.add_argument("--period", required=True, type=str, help="period")
    args = parser.parse_args()
    return args


def load_processes(period):
    filepath = os.path.join(os.environ['ANALYSIS_PATH'], 'config', period, 'processes.yaml')
    with open(filepath, "r") as f:
        processes = yaml.safe_load(f)
    return processes


def load_configs(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    with open(config_dict["meta_data"]["global_config"], "r") as f:
        global_config = yaml.safe_load(f)
    return config_dict, global_config


def process_datasets(period, group_name, group_data, global_config, meta_data, output_columns):        
    # List to hold the RDataFrames for this high-level group
    print(f"\n--- Starting Processing for Group: {group_name} ---")
    for dataset_name in group_data['datasets']:
        print(f"\t On dataset {dataset_name}")
        output_filename = os.path.join(meta_data['output_folder'], period, f"{dataset_name}.root")
        pattern = os.path.join(meta_data['input_folder'], period, dataset_name, "*.root")
        filelist = glob(pattern)
        print(filelist)
        rdf = ROOT.RDataFrame("Events", filelist)
        dfw = analysis.DataFrameBuilderForHistograms(rdf, global_config, period)
        dfw = analysis.PrepareDfForVBFNetworkInputs(dfw)
        rdf = dfw.df
        # Add a column defining the specific dataset name
        # Note: We must use C++ string literal syntax, hence the extra quotes
        rdf = rdf.Define("dataset", f'(std::string)"{dataset_name}"')
        rdf = rdf.Define("process", f'(std::string)"{group_name}"')
        rdf = rdf.Define("era", f'(std::string)"{period}"')
        # Do selection/filtering
        rdf = rdf.Filter("baseline")
        rdf = rdf.Filter("FilteredJet_pt.size() >= 2")
        # Save the result
        #save_column_names = ROOT.std.vector("string")(output_columns)
        rdf.Snapshot("Events", output_filename, output_columns)
        del rdf
 

if __name__ == '__main__':

    # Init from args and configs
    args = get_args()
    config, global_config  = load_configs(args.config)
    output_columns = [
        "FullEventId",
        "isData",
        "weight_MC_Lumi_pu",
        "era",
        "process",
        "dataset",
        "baseline",
        "Signal_Fit",
        "Signal_ext",
        "H_sideband",
        "Z_sideband",
        "FilteredJet_pt",
        "FilteredJet_eta",
        "FilteredJet_phi",
        "FilteredJet_btagPNetQvG",
        "FilteredJet_puIdDisc"
    ]


    if args.period.lower() == 'all':
        eras = ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
    else:
        eras = [args.period]

    for period in eras:
        print(f"\n***** Starting Processing for Era: {period} *****")
        process_config = load_processes(period)
        for sample_type in config['sample_list']:
            if sample_type == 'data':
                group_data = {'datasets' : ['data']}
            else:
                group_data = process_config[sample_type]
            process_datasets(
                    period=period, 
                    group_name=sample_type, 
                    group_data=group_data, 
                    global_config=global_config, 
                    meta_data=config['meta_data'], 
                    output_columns=output_columns
                )
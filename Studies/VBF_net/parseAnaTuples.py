
import os
from FLAF.Common.Setup import Setup
from Corrections.Corrections import Corrections
import argparse
from glob import glob
from pprint import pprint
from Analysis.histTupleDef import *

import ROOT
import yaml
import uproot

import Analysis.H_mumu as hmumu
from FLAF.Common.Setup import Setup

from columns_config_union import columns_config

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

def load_configs(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    with open(config_dict["meta_data"]["global_config"], "r") as f:
        global_config = yaml.safe_load(f)
    return config_dict, global_config


def process_datasets(setup, group_name, group_data, meta_data, output_columns, selection_cut=None):        
    print(f"\n--- Starting Processing for Group: {group_name} ---")

    # Parse from setup
    global_config = setup.global_params
    global_config['process_name'] = group_name
    period = setup.period
    unc_cfg_dict=setup.weights_config
    hist_cfg_dict = setup.hists

    for dataset_name in group_data['datasets']:

        # Per dataset setup modification
        process_group = setup.datasets[dataset_name]["process_group"]
        process_name = setup.datasets[dataset_name]["process_name"]
        setup.global_params["process_name"] = process_name
        setup.global_params["process_group"] = process_group

        # Locate the actual anaTuple files
        print(f"\t On dataset {dataset_name}")
        output_filename = os.path.join(meta_data['output_folder'], period, f"{dataset_name}.root")
        pattern = os.path.join(meta_data['input_folder'], period, dataset_name, "*.root")
        filelist = glob(pattern)
        if not filelist:
            print("******* WARNING: empty anaTuples:", dataset_name)
            continue

        # Init corrections
        if group_name == 'data':
            is_data = True
        else:
            is_data = False
        hmumu.InitializeCorrections(period, dataset_name)
        corrections = Corrections.getGlobal()

        # Init DataFrameBuilder and add analysis variables
        rdf = ROOT.RDataFrame("Events", filelist)
        dfw = hmumu.DataFrameBuilderForHistograms(rdf, global_config, period, corrections)
        dfw = hmumu.PrepareDFBuilder(dfw)
        dfw = hmumu.PrepareDfForVBFNetworkInputs(dfw)
        dfw = DefineWeightForHistograms(
            dfw=dfw,
            isData=is_data,
            uncName="Central",
            uncScale="Central",
            unc_cfg_dict=unc_cfg_dict,
            hist_cfg_dict=hist_cfg_dict,
            global_params=global_config,
            final_weight_name="total_weight",
            df_is_central=True
        )

        # Add a column defining the specific dataset name
        rdf = dfw.df
        rdf = rdf.Define("dataset", f'(std::string)"{dataset_name}"')
        rdf = rdf.Define("process", f'(std::string)"{group_name}"')
        rdf = rdf.Define("era", f'(std::string)"{period}"')

        # Do selection/filtering
        rdf = rdf.Filter("baseline_muonJet")
        rdf = rdf.Filter("FilteredJet_pt_vec.size() >= 2")
        if meta_data['selection_cut']:
            cut = meta_data['selection_cut']
            rdf = rdf.Filter(cut)

        # Save the result
        rdf.Snapshot("Events", output_filename, output_columns)
        del rdf
 

if __name__ == '__main__':

    # Init from args and configs
    args = get_args()
    config, global_config  = load_configs(args.config)
    output_columns = columns_config['metadata'] + columns_config['flat_vars'] + columns_config['jet_vars']

    if args.period.lower() == 'all':
        eras = ["Run3_2022", "Run3_2022EE", "Run3_2023", "Run3_2023BPix"]
    else:
        eras = [args.period]

    for period in eras:
        print(f"\n***** Starting Processing for Era: {period} *****")
        setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], period, None)
        setup.global_params["compute_rel_weights"] = False
        analysis_setup(setup)
        for sample_type in config['sample_list']:
            global_config['process_name'] = sample_type
            if sample_type == 'data':
                group_data = {'datasets' : ['data']}
            elif sample_type == 'DY':
                group_data = {'datasets' : ['DYto2Mu_MLL_105to160_amcatnloFXFX']}
            else:
                group_data = setup.base_processes[sample_type]
            process_datasets(
                    setup=setup, 
                    group_name=sample_type, 
                    group_data=group_data, 
                    meta_data=config['meta_data'], 
                    output_columns=output_columns,
                )

    # Run through outputs, delete empties
    # print("### Running over outputs, deleting empty root files...")
    # print("Switching to output dir:", config['meta_data']['output_folder'])
    # os.chdir(config['meta_data']['output_folder'])
    # filelist = glob('*/*.root')
    # for filename in filelist:
    #     f = ROOT.TFile(filename)
    #     tree = f.Get("Events")
    #     count = tree.GetEntries()
    #     print("\t", filename, count)
    #     if count == 0:
    #         print("\t\t REMOVING", filename)
    #         os.remove(filename)
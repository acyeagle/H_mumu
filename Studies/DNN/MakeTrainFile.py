import argparse
import os
import sys
from glob import glob
from math import ceil

import psutil
import ROOT
import yaml
import uproot

from model_generation.parse_column_names import parse_column_names

import Analysis.H_mumu as analysis
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup


ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()
from FLAF.Common.Utilities import DeclareHeader


def get_entries_per_batch(directory, nBatches):
    total = 0
    pattern = os.path.join(directory, "*.root")
    for filename in glob(pattern):
        with uproot.open(filename) as f:
            total += f["Events"].num_entries
    return ceil(total / nBatches)


def create_file(
    config_dict, global_cfg_dict, general_cfg_dict, period, out_filename, dataset
):
    print(f"Starting create file. Memory usage in MB is {get_memory_usage()}")

    # Set the batch params for this anaTuple directory
    nBatches = config_dict["meta_data"]["nBatches"]
    nEntriesPerBatch = get_entries_per_batch(dataset, nBatches)
    nBatchStart = 0
    nBatchEnd = nBatchStart + nEntriesPerBatch

    step_idx = 0

    # Get the name/type (And order!) of signal columns
    master_column_names = []
    master_column_types = []
    master_column_names_vec = ROOT.std.vector("string")()

    # Init temp
    tmp_dir = config_dict["meta_data"]["temp_folder"]
    tmp_filename = os.path.join(tmp_dir, f"tmp{step_idx}.root")
    tmpnext_filename = os.path.join(tmp_dir, f"tmp{step_idx+1}.root")

    # Load in the RDataFrame
    pattern = os.path.join(dataset, "*.root")
    process_filelist = glob(pattern)
    df_in = ROOT.RDataFrame("Events", process_filelist)

    # Filter for nLeps and Parity (iterate cut in config)
    if "iterate_cut" in config_dict["meta_data"].keys():
        df_in = df_in.Filter(config_dict["meta_data"]["iterate_cut"])

    # Load df_out, if first iter then load an empty, otherwise load the past file
    if step_idx == 0:
        df_out = ROOT.RDataFrame(nBatches * nEntriesPerBatch)
        df_out = df_out.Define("is_valid", "false")

        # Fill master column nametype
        master_column_names = df_in.GetColumnNames()
        master_column_types = [
            str(df_in.GetColumnType(str(c))) for c in master_column_names
        ]
        for name in master_column_names:
            master_column_names_vec.push_back(name)
    else:
        df_out = ROOT.RDataFrame("Events", tmp_filename)

    local_column_names = df_in.GetColumnNames()
    local_column_types = [str(df_in.GetColumnType(str(c))) for c in local_column_names]
    local_column_names_vec = ROOT.std.vector("string")()
    for name in local_column_names:
        local_column_names_vec.push_back(name)

    # Need a local_to_master_map so that local columns keep the same index as the master columns
    local_to_master_map = [
        list(master_column_names).index(local_name) for local_name in local_column_names
    ]
    master_size = len(master_column_names)

    queue_size = 10
    max_entries = nEntriesPerBatch * nBatches

    tuple_maker = ROOT.analysis.TupleMaker(*local_column_types)(queue_size, max_entries)

    df_out = tuple_maker.FillDF(
        ROOT.RDF.AsRNode(df_out),
        ROOT.RDF.AsRNode(df_in),
        local_to_master_map,
        master_size,
        local_column_names_vec,
        nBatchStart,
        nBatchEnd,
        nEntriesPerBatch,
    )

    for column_idx, column_name in enumerate(master_column_names):
        column_type = master_column_types[column_idx]

        if step_idx == 0:
            df_out = df_out.Define(
                str(column_name),
                f"_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_type}() ",
            )
        else:
            if column_name not in local_column_names:
                continue
            df_out = df_out.Redefine(
                str(column_name),
                f"_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_name} ",
            )

    df_out = df_out.Redefine("is_valid", "(is_valid) || (_entry)")

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    # snapshotOptions.fOverwriteIfExists=False
    # snapshotOptions.fLazy=True
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, "k" + "ZLIB")
    snapshotOptions.fCompressionLevel = 4
    ROOT.RDF.Experimental.AddProgressBar(df_out)
    print("Going to snapshot")
    save_column_names = ROOT.std.vector("string")(master_column_names)
    save_column_names.push_back("is_valid")
    df_out.Snapshot("Events", tmpnext_filename, save_column_names, snapshotOptions)

    tuple_maker.join()

    step_idx += 1

    os.system(f"rm {tmp_filename}")

    print("Finished create file loop, now we must add the DNN variables")
    tmp_dir = config_dict["meta_data"]["temp_folder"]
    tmp_filename = os.path.join(tmp_dir, f"tmp{step_idx}.root")
    tmpnext_filename = os.path.join(tmp_dir, f"tmp{step_idx+1}.root")

    df_out = ROOT.RDataFrame("Events", tmp_filename)

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    # snapshotOptions.fOverwriteIfExists=False
    # snapshotOptions.fLazy=True
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, "k" + "ZLIB")
    snapshotOptions.fCompressionLevel = 4
    ROOT.RDF.Experimental.AddProgressBar(df_out)
    print("Going to snapshot")
    # Only need to save the prexisting columns plus the new DNN variables
    # to add kwargset for isData
    dfw_out = analysis.DataFrameBuilderForHistograms(df_out, global_cfg_dict, period)
    dfw_out = analysis.PrepareDFBuilder(dfw_out)
    dfw_out.colToSave += [c for c in df_out.GetColumnNames()]
    col_to_save = AddColumnsToSave(general_cfg_dict)

    dfw_out.df.Snapshot(
        "Events", tmpnext_filename, Utilities.ListToVector(col_to_save), snapshotOptions
    )

    print(f"Finished create file, will copy {tmpnext_filename} file to final output {out_filename}")
    os.system(f"cp {tmpnext_filename} {out_filename}")
    os.system(f"rm {tmp_filename}")
    os.system(f"rm {tmpnext_filename}")


def load_framework():
    sys.path.append(os.environ["ANALYSIS_PATH"])
    ana_path = os.environ["ANALYSIS_PATH"]
    # Include from ANALYSIS_PATH
    for header in [
        "FLAF/include/Utilities.h",
        "include/Helper.h",
        "include/HmumuCore.h",
        "FLAF/include/AnalysisTools.h",
        "FLAF/include/AnalysisMath.h",
    ]:
        DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)
    # Include from Studies/DNN local include
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    headers = ["include/TupleMaker.h"]
    for header in headers:
        header_path = os.path.join(headers_dir, header)
        ROOT.gInterpreter.Declare(f'#include "{header_path}"')
    


def get_args():
    parser = argparse.ArgumentParser(description="Create train/test file(s) for DNN.")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="The configuration yaml specifiying the samples and batches",
    )
    parser.add_argument("--period", required=True, type=str, help="period")
    args = parser.parse_args()
    return args


def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info()[0] / float(2**20)


def yaml_to_config(yamlname):
    with open(yamlname, "r") as file:
        config_dict = yaml.safe_load(file)
    glb_cfg_dict_name = config_dict["meta_data"]["global_config"]
    with open(glb_cfg_dict_name, "r") as glb_cfg_file:
        global_cfg_dict = yaml.safe_load(glb_cfg_file)
    general_cfg_dict_name = config_dict["meta_data"]["general_config"]
    with open(general_cfg_dict_name, "r") as general_cfg_dict_file:
        general_cfg_dict = yaml.safe_load(general_cfg_dict_file)

    return config_dict, global_cfg_dict, general_cfg_dict


if __name__ == "__main__":

    load_framework()
    args = get_args()

    config_dict, global_cfg_dict, general_cfg_dict = yaml_to_config(args.config)

    for dataset in config_dict["dataset_list"]:
        anaTuples_directory = os.path.join(
            config_dict["meta_data"]["input_folder"], dataset
        )
        output_name = os.path.join(config_dict["meta_data"]["output_folder"], dataset)
        output_name += ".root"
        create_file(
            config_dict,
            global_cfg_dict,
            general_cfg_dict,
            args.period,
            output_name,
            anaTuples_directory,
        )

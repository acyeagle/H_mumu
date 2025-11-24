import importlib
from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


initialized = False
analysis = None


def Initialize():
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gInterpreter.Declare(
            f'#include "FLAF/include/pnetSF.h"'
        )  # do we need this??
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
        ROOT.gInterpreter.Declare(
            f'#include "include/Helper.h"'
        )  # not related to FullEvtId definition but needed for analysis specific purpose. At a certain point it will be moved to analysis specific section.
        initialized = True


def analysis_setup(setup):
    global analysis
    analysis_import = setup.global_params["analysis_import"]
    analysis = importlib.import_module(f"{analysis_import}")



def GetDfw(
    df,
    df_caches,
    global_params,
    shift="Central",
    col_names_central=[],
    col_types_central=[],
    cache_map_name="cache_map_Central",
):
    period = global_params["era"]
    kwargset = (
        {}
    )  # here go the customisations for each analysis eventually extrcting stuff from the global params
    kwargset["isData"] = global_params["process_group"] == "data"
    kwargset["wantTriggerSFErrors"] = global_params["compute_rel_weights"]
    kwargset["colToSave"] = []

    
    dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period, **kwargset)

    if df_caches:
        k = 0

        for df_cache in df_caches:
            dfWrapped_cache = analysis.DataFrameBuilderForHistograms(
                df_cache, global_params, period, **kwargset
            )
            AddCacheColumnsInDf(dfw, dfWrapped_cache, f"{cache_map_name}_{k}")
            k += 1

    if shift == "Valid" and global_params["compute_unc_variations"]:
        dfw.CreateFromDelta(col_names_central, col_types_central)
    if shift != "Central" and global_params["compute_unc_variations"]:
        dfw.AddMissingColumns(col_names_central, col_types_central)
    new_dfw = analysis.PrepareDfForHistograms(dfw)
    if global_params["further_cuts"]:
        for key in global_params["further_cuts"].keys():
            vars_to_add = global_params["further_cuts"][key][0]
            for var_to_add in vars_to_add:
                if var_to_add not in new_dfw.colToSave:
                    new_dfw.colToSave.append(var_to_add)
    return new_dfw


# def GetDfw(
#     df,
#     df_cache,
#     global_params,
#     shift="Central",
#     col_names_central=[],
#     col_types_central=[],
#     cache_map_name="cache_map_Central",
# ):
#     period = global_params["era"]
#     kwargset = (
#         {}
#     )  # here go the customisations for each analysis eventually extrcting stuff from the global params
#     kwargset["isData"] = global_params["process_group"] == "data"
#     kwargset["wantTriggerSFErrors"] = global_params["compute_rel_weights"]
#     kwargset["colToSave"] = []

#     dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period, **kwargset)

#     if df_cache:

#         dfWrapped_cache = analysis.DataFrameBuilderForHistograms(
#             df_cache, global_params, **kwargset
#         )
#         AddCacheColumnsInDf(dfw, dfWrapped_cache, cache_map_name)
#     if shift == "Valid" and global_params["compute_unc_variations"]:
#         dfw.CreateFromDelta(col_names_central, col_types_central)
#     if shift != "Central" and global_params["compute_unc_variations"]:
#         dfw.AddMissingColumns(col_names_central, col_types_central)
#     new_dfw = analysis.PrepareDfForHistograms(dfw)
#     return new_dfw


def DefineWeightForHistograms(
    dfw,
    uncName,
    uncScale,
    unc_cfg_dict,
    hist_cfg_dict,
    global_params,
    final_weight_name="weight_for_hists",
):
    categories = global_params["categories"]
    process_group = global_params["process_group"]
    process_group = global_params["process_group"]
    isCentral = uncName == "Central"
    total_weight_expression = (
        analysis.GetWeight("muMu") if process_group != "data" else "1"
    )  # are we sure?
    weight_name = "final_weight"
    if weight_name not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name, total_weight_expression)
    if not isCentral:  # and type(unc_cfg_dict['norm']) == dict:
        if (
            uncName in unc_cfg_dict["norm"].keys()
            and "expression" in unc_cfg_dict["norm"][uncName].keys()
        ):
            weight_name = unc_cfg_dict["norm"][uncName]["expression"].format(
                scale=uncScale
            )
    dfw.df = dfw.df.Define(final_weight_name, weight_name)

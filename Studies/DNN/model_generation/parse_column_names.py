from itertools import chain
from pprint import pprint


def parse_column_names(config, column_type="all"):
    """
    Takes a config file specifying which columns to save in the NN sample sets.
    Generally this would be ds_setup/general.yaml.
    Returns a list of all the (unique) column names.
    """

    Muon_vars = []
    for mu_idx in [1, 2]:
        for mu_var in config["Muon"]:
            Muon_vars.append(mu_var.format(mu_idx))

    VBFJet_vars = []
    for j_idx in [1, 2]:
        for VBFJ_var in config["VBFJet"]:
            VBFJet_vars.append(f"j{j_idx}_{VBFJ_var}")
    # SelectedJet_vars = config['vars_to_save']['SelectedJet']
    # for Jvar in SelectedJet_vars:
    #     col_to_save.append(f"SelectedJet_{Jvar}")

    SoftJet_vars = []
    try:
        for SJvar in config["SoftJet"]:
            SoftJet_vars.append(f"SoftJet_{SJvar}")
    except KeyError:
        pass

    VBFJetPair_vars = config["VBFJetPair"]
    MuJet_vars = config["MuJet"]
    MuPair_vars = config["MuPair"]
    Global_vars = config["Global"]
    Category_vars = config["Categories"]
    Region_vars = config["Regions"]
    Sign_vars = config["Sign"]
    # nJet_vars = config["nJets"]
    Weight_vars = config["Weight"]

    data_columns = [
        Muon_vars,
        MuPair_vars,
        MuJet_vars,
        VBFJet_vars,
        VBFJetPair_vars,
        SoftJet_vars,
    ]

    header_columns = [
        Global_vars,
        Weight_vars,
    ]

    selection_columns = [Sign_vars, Region_vars, Category_vars]

    data_columns = [x for x in data_columns if x is not None]
    header_columns = [x for x in header_columns if x is not None]
    selection_columns = [x for x in selection_columns if x is not None]

    data_columns = list(chain.from_iterable(data_columns))
    header_columns = list(chain.from_iterable(header_columns))
    selection_columns = list(chain.from_iterable(selection_columns))

    print("Data columns:")
    pprint(data_columns)
    print("Header columns:")
    pprint(header_columns)
    print("Selection columns:")
    pprint(selection_columns)

    if column_type == "all":
        return header_columns + selection_columns + data_columns
    elif column_type == "data":
        return data_columns
    elif column_type == "header":
        return header_columns
    elif column_type == "selection":
        return selection_columns

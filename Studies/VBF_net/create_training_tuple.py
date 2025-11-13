import os
from glob import glob

import ROOT as root

base_dir = os.getcwd()
os.chdir("/eos/user/a/ayeagle/H_mumu/root_files/v3_parse1/")

# Define the output columns, filter mask, and files to use

filelist = []
sample_patterns = [
    "GluGluHto2Mu.root",
    "VBFHto2Mu.root",
    "EWK_2L2J*.root",
    "DYto2*.root",
]
for pattern in sample_patterns:
    filelist += glob("*/" + pattern)

cols = [
    "FullEventId",
    "isData",
    "weight_MC_Lumi_pu",
    "era",
    "process",
    "dataset",
    "Signal_Fit",
    "H_sideband",
    "Z_sideband",
    # 'mu1_eta',
    # 'mu1_phi',
    # 'mu2_eta',
    # 'mu2_phi',
    "FilteredJet_pt",
    "FilteredJet_eta",
    "FilteredJet_phi",
]

mask_expression = (
    "(Jet_pt > 25) && "
    "((ROOT::VecOps::abs(Jet_eta) < 2.5) || (ROOT::VecOps::abs(Jet_eta) > 3.0) || (Jet_pt > 50))"
)

# Load the RDF and filter
rdf = root.RDataFrame("Events", filelist)
rdf = rdf.Filter("Signal_ext")
rdf = rdf.Define("FilteredJet_mask", mask_expression)
rdf = rdf.Filter("ROOT::VecOps::Any(FilteredJet_mask)")
rdf = (
    rdf.Define("FilteredJet_pt", "Jet_pt[FilteredJet_mask]")
    .Define("FilteredJet_eta", "Jet_eta[FilteredJet_mask]")
    .Define("FilteredJet_phi", "Jet_phi[FilteredJet_mask]")
)
rdf = rdf.Filter("FilteredJet_pt.size() >= 2")

# os.chdir(base_dir)
# Save the output to a root file
rdf.Snapshot("Events", "VBFNet_samples_mk2.root", cols)

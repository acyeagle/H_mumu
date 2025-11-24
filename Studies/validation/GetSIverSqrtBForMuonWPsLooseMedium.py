import ROOT
import sys
import os
import importlib
import glob
if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])

import argparse
import FLAF.Common.Utilities as Utilities
from FLAF.Common.Setup import Setup
# import importlib
from FLAF.Common.HistHelper import *
from Analysis.H_mumu import *
ROOT.EnableImplicitMT()




parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True, type=str)
args = parser.parse_args()

period = f"Run3_{args.year}"
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


setup = Setup.getGlobal(os.environ["ANALYSIS_PATH"], period, "")
analysis_import = setup.global_params["analysis_import"]
analysis = importlib.import_module(f"{analysis_import}")

global_params_sig = setup.global_params
global_params_sig["process_name"] = setup.samples["VBFHto2Mu"]["process_name"]
global_params_sig["process_group"] = setup.samples["VBFHto2Mu"]["process_group"]

global_params_bckg = setup.global_params
global_params_bckg["process_name"] = "DY"
global_params_bckg["process_group"] = "DY"


kwargset = (
    {}
)  # here go the customisations for each analysis eventually extrcting stuff from the global params

kwargset["isData"] = False
kwargset["wantTriggerSFErrors"] = False
kwargset["colToSave"] = []


# /eos/user/e/eusebi/Hmumu/anaTuples/
sig_list= [
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/GluGluHto2Mu/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/VBFHto2Mu/anaTuple*.root"]
full_sig_list = [
    filename
    for pattern in sig_list
    for filename in glob.glob(pattern)
]

print(f"Trovati {len(full_sig_list)} file totali.")
print("Esempi di file trovati:")
for file in full_sig_list[:5]:
    print(f"- {file}")

bckg_list = [
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2E_M_50_0J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2E_M_50_1J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2E_M_50_2J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2Mu_M_50_0J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2Mu_M_50_1J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2Mu_M_50_2J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2Tau_M_50_0J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2Tau_M_50_1J_amcatnloFXFX/anaTuple*.root",
    f"/eos/user/e/eusebi/Hmumu/anaTuples/v4_20Nov_NewJetIDs_LooseMuons/Run3_{args.year}/DYto2Tau_M_50_2J_amcatnloFXFX/anaTuple*.root"
]



full_bckg_list = [
    filename
    for pattern in bckg_list
    for filename in glob.glob(pattern)
]

# Stampa i primi N file per verifica
print(f"Trovati {len(full_bckg_list)} file totali.")
print("Esempi di file trovati:")
for file in full_bckg_list[:5]:
    print(f"- {file}")

rdf_signal = ROOT.RDataFrame("Events",Utilities.ListToVector(full_sig_list))
rdf_bckg = ROOT.RDataFrame("Events",Utilities.ListToVector(full_bckg_list))
dfw_sig = analysis.DataFrameBuilderForHistograms(rdf_signal, global_params_sig, period, **kwargset)
dfw_bckg = analysis.DataFrameBuilderForHistograms(rdf_bckg, global_params_bckg, period, **kwargset)

for dfw in [dfw_sig,dfw_bckg]:
    dfw.df = Utilities.defineP4(dfw.df, f"mu1")
    dfw.df = Utilities.defineP4(dfw.df, f"mu2")
    dfw.df = dfw.df.Define("m_mumu", f"(mu1_p4 + mu2_p4).M()")
    dfw.defineChannels()
    dfw.defineTriggers()
    # dfw.df = AddNewDYWeights(dfw.df, dfw.period, False)
    # dfw.df = AddMuTightIDWeights(dfw.df, dfw.period)
    # dfw.df = dfw.df.Define("Jet_vetoMap","Jet_pt > 15 && ( Jet_passJetIdTightLepVeto ) && (Jet_chEmEF + Jet_neEmEF < 0.9) && Jet_isInsideVetoRegion")
    dfw.df = JetCollectionDef(dfw.df)
    dfw.df = VBFJetSelection(dfw.df)
    dfw.SignRegionDef()
    dfw.defineRegions()
    dfw.defineCategories()

baseline = "OS && trigger_sel && JetTagSel"
category_to_select = "mu1_pfRelIso04_all < {0} && mu2_pfRelIso04_all < {0}" # no pT cut on muons - default is 10

mass_region = "Signal_Fit"
VBF_veto_region = f"muons_IsoWP && HasVBF && j1_pt >= 35 && j2_pt >= 25 && HornJetVeto_def"
ggH_region = f"muons_IsoWP && !VBF_veto_region"

for WP in [0.25, 0.2]:
    for Id_WP in ["loose", "medium"]:
        category_to_select_ID = "mu1_{0}Id  && mu1_{0}Id".format(Id_WP)
        category_to_select_WP = category_to_select.format(WP)
        print("************************")
        print( "considering WP = ", WP)
        print( "considering id = ", Id_WP)
        print("************************")

        bckg_baseline_before_isoCut = dfw_bckg.df.Define(f"muJet_baseline", baseline).Filter("muJet_baseline").Count().GetValue()
        bckg_baseline_before_isoCut_inSigMassRegion = dfw_bckg.df.Define(f"muJet_baseline", baseline).Filter(f"muJet_baseline && {mass_region}").Count().GetValue()
        bckg_baseline = dfw_bckg.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Filter("muons_IsoWP").Count().GetValue()
        bckg_baseline_inSigMassRegion = dfw_bckg.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Filter(f"muons_IsoWP  && {mass_region}").Count().GetValue()


        bckg_ggH = dfw_bckg.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Define("VBF_veto_region",VBF_veto_region).Define("ggH_region", ggH_region).Filter(f"{mass_region} && ggH_region").Count().GetValue()
        bckg_VBF = dfw_bckg.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Define("VBF_veto_region",VBF_veto_region).Filter(f"{mass_region} && VBF_veto_region").Count().GetValue()

        print("On DY Backgrounds ")
        print(f"for WP = {WP},before any iso cut = {bckg_baseline_before_isoCut}, after cutting on iso & Id= {bckg_baseline}, bckg ggH = {bckg_ggH}, bckg VBF = {bckg_VBF}")
        print(f"background efficiency: baseline = {bckg_baseline_inSigMassRegion/bckg_baseline_before_isoCut_inSigMassRegion}, ggH = {bckg_ggH/bckg_baseline_before_isoCut_inSigMassRegion}, VBF = {bckg_VBF/bckg_baseline_before_isoCut_inSigMassRegion}")
        print(f"background efficiency w.r.t. to entire phase space: baseline(all mass range) = {bckg_baseline/bckg_baseline_before_isoCut}, baseline = {bckg_baseline_inSigMassRegion/bckg_baseline_before_isoCut}, ggH = {bckg_ggH/bckg_baseline_before_isoCut}, VBF = {bckg_VBF/bckg_baseline_before_isoCut}")

        sig_baseline_before_isoCut = dfw_sig.df.Define(f"muJet_baseline", baseline).Filter("muJet_baseline").Count().GetValue()
        sig_baseline_before_isoCut_inSigMassRegion = dfw_sig.df.Define(f"muJet_baseline", baseline).Filter(f"muJet_baseline && {mass_region}").Count().GetValue()
        sig_baseline = dfw_sig.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Filter("muons_IsoWP").Count().GetValue()
        sig_baseline_inSigMassRegion = dfw_sig.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Filter(f"muons_IsoWP  && {mass_region}").Count().GetValue()

        print()
        sig_ggH = dfw_sig.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Define("VBF_veto_region",VBF_veto_region).Define("ggH_region", ggH_region).Filter(f"{mass_region} && ggH_region").Count().GetValue()
        sig_VBF = dfw_sig.df.Define(f"muJet_baseline", baseline).Define("muons_IsoWP", f"muJet_baseline && {category_to_select_WP} && {category_to_select_ID}").Define("VBF_veto_region",VBF_veto_region).Filter(f"{mass_region} && VBF_veto_region").Count().GetValue()

        print("On signals ")
        print(f"for WP = {WP},before any iso cut = {sig_baseline_before_isoCut}, after cutting on iso & Id= {sig_baseline}, sig ggH = {sig_ggH}, sig VBF = {sig_VBF}")
        print(f"background efficiency: baseline = {sig_baseline_inSigMassRegion/sig_baseline_before_isoCut_inSigMassRegion}, ggH = {sig_ggH/sig_baseline_before_isoCut_inSigMassRegion}, VBF = {sig_VBF/sig_baseline_before_isoCut_inSigMassRegion}")
        print(f"background efficiency w.r.t. to entire phase space: baseline(all mass range) = {sig_baseline/sig_baseline_before_isoCut}, baseline = {sig_baseline_inSigMassRegion/sig_baseline_before_isoCut}, ggH = {sig_ggH/sig_baseline_before_isoCut}, VBF = {sig_VBF/sig_baseline_before_isoCut}")
        print()



        eff_final_baseline_before_isoCut = sig_baseline_before_isoCut/math.sqrt(bckg_baseline_before_isoCut)
        eff_final_baseline_before_isoCut_inSigMassRegion = sig_baseline_before_isoCut_inSigMassRegion/math.sqrt(bckg_baseline_before_isoCut_inSigMassRegion)
        eff_final_baseline = sig_baseline/math.sqrt(bckg_baseline)
        eff_final_baseline_inSigMassRegion = sig_baseline_inSigMassRegion/math.sqrt(bckg_baseline_inSigMassRegion)
        eff_final_ggH = sig_ggH/math.sqrt(bckg_ggH)
        eff_final_VBF = sig_VBF/math.sqrt(bckg_VBF)

        print(f"\n for WP = {WP}\n s_sqrtB(before iso cut, all phase space) = {eff_final_baseline_before_isoCut} \ns_sqrtB(before iso cut, sig mass region) = {eff_final_baseline_before_isoCut_inSigMassRegion} \ns_sqrtB(baseline selection, all phase space) = {eff_final_baseline}\ns_sqrtB(baseline selection, sig mass region) = {eff_final_baseline_inSigMassRegion}\ns_sqrtB(ggH) = {eff_final_ggH}\ns_sqrtB(VBF) = {eff_final_VBF}")
        print()
        print()

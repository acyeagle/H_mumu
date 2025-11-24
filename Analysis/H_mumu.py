import ROOT
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *
from Analysis.GetTriggerWeights import *
from Analysis.CorrectionsRelatedFunctions import *
from Analysis.MuonRelatedFunctions import *
from Analysis.JetRelatedFunctions import *

for header in [
    "FLAF/include/Utilities.h",
    "include/Helper.h",
    "include/HmumuCore.h",
    "FLAF/include/AnalysisTools.h",
    "FLAF/include/AnalysisMath.h",
]:
    DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)



def createKeyFilterDict(global_params, period):
    filter_dict = {}
    filter_str = ""
    channels_to_consider = global_params["channels_to_consider"]
    # sign_regions_to_consider = global_params["MuMuMassRegions"]
    categories = global_params["categories"]

    ### add custom categories eventually:
    custom_categories = []
    custom_categories_name = global_params.get(
        "custom_categories", None
    )  # can be extended to list of names
    if custom_categories_name:
        custom_categories = list(global_params.get(custom_categories_name, []))
        if not custom_categories:
            print("No custom categories found")

    ### regions
    custom_regions = []
    custom_regions_name = global_params.get(
        "custom_regions", None
    )  # can be extended to list of names, if for example adding QCD regions + other control regions
    if custom_regions_name:
        custom_regions = list(global_params.get(custom_regions_name, []))
        if not custom_regions:
            print("No custom regions found")

    all_categories = categories + custom_categories
    custom_subcategories = list(global_params.get("custom_subcategories", []))
    triggers_dict = global_params["hist_triggers"]
    for ch in channels_to_consider:
        triggers = triggers_dict[ch]["default"]
        if period in triggers_dict[ch].keys():
            triggers = triggers_dict[ch][period]
        for reg in custom_regions:
            for cat in all_categories:
                filter_base = f" ( {ch} && {triggers} && {reg} && {cat} ) "
                if custom_subcategories:
                    for subcat in custom_subcategories:
                        # filter_base += f"&& {custom_subcat}"
                        filter_str = f"(" + filter_base + f" && {subcat}"
                        filter_str += ")"
                        key = (ch, reg, cat, subcat)
                        filter_dict[key] = filter_str
                else:
                    filter_str = f"(" + filter_base
                    filter_str += ")"
                    key = (ch, reg, cat)
                    filter_dict[key] = filter_str
    return filter_dict


def GetBTagWeight(global_cfg_dict, cat, applyBtag=False):
    btag_weight = "1"
    btagshape_weight = "1"
    global_cfg_dict["ApplyBweight"]
    if global_cfg_dict["ApplyBweight"]:
        if applyBtag:
            if global_cfg_dict["btag_wps"][cat] != "":
                btag_weight = f"weight_bTagSF_{btag_wps[cat]}_Central"
        else:
            if cat not in global_cfg_dict["boosted_categories"] and not cat.startswith(
                "baseline"
            ):
                btagshape_weight = "weight_bTagShape_Central"
    return f"{btag_weight}*{btagshape_weight}"


def SaveVarsForNNInput(variables):
    mumu_vars = [
        "pt_mumu",
        "y_mumu",
        "eta_mumu",
        "phi_mumu",
        "m_mumu",
        "dR_mumu",
        "cosTheta_CS",
        "mu1_pt_rel",
        "mu2_pt_rel",
        "mu1_eta",
        "mu2_eta",
        "phi_CS",
    ]  # , "Ebeam"
    jj_vars = [
        "j1_pt",
        "j1_eta",
        "j2_pt",
        "j2_eta",
        "HasVBF",
        "m_jj",
        "delta_eta_jj",
    ]  # ,"j1_idx","j1_y","j1_phi","delta_phi_jj"
    mumu_jj_vars = [
        "Zepperfield_Var",
        "R_pt",
        "pt_centrality",
        "minDeltaPhi",
        "minDeltaEta",
        "minDeltaEtaSigned",
    ]  # , "pT_all_sum","pT_jj_sum",
    softJets_vars = [
        "N_softJet",
        "SoftJet_energy",
        "SoftJet_Et",
        "SoftJet_HtCh_fraction",
        "SoftJet_HtNe_fraction",
        "SoftJet_HtHF_fraction",
    ]  # ATTENTION: THESE ARE VECTORS, NOT FLAT OBSERVABLES
    global_vars = [
        "entryIndex",
        "luminosityBlock",
        "run",
        "event",
        "sample_type",
        "sample_name",
        "period",
        "isData",
        "nJet",
    ]  # ,"PV_npvs"
    # global_vars = ["FullEventId","luminosityBlock", "run","event", "sample_type", "sample_name", "period", "isData", "nJet"] # ,"PV_npvs"
    for var in global_vars + mumu_vars + jj_vars + mumu_jj_vars + softJets_vars:
        variables.append(var)
    return variables


def GetWeight(channel="muMu"):
    weights_to_apply = [
        "weight_MC_Lumi_pu",
        "weight_XS",
        "newDYWeight_ptLL_nano"
        # "weight_DYw_DYWeightCentral",
        # "weight_EWKCorr_VptCentral",
    ]  # ,"weight_EWKCorr_ewcorrCentral"] #

    trg_weights_dict = {
        # "muMu": ["weight_trigSF_singleMu"]
        "muMu": ["weight_trigSF_singleMu_mediumID_mediumIso"]
        # "muMu": ["weight_trigSF_singleMu_tightID_tightIso"]

    }
    # ID_weights_dict = {
    #     "muMu": [
    #         # "weight_mu1_HighPt_MuonID_SF_MediumIDCentral",
    #         # "weight_mu1_LowPt_MuonID_SF_MediumIDCentral",
    #         # "weight_mu1_MuonID_SF_LoosePFIsoCentral",
    #         "weight_mu1_MuonID_SF_MediumID_TrkCentral",
    #         "weight_mu1_MuonID_SF_MediumIDLoosePFIsoCentral",
    #         # "weight_mu2_HighPt_MuonID_SF_MediumIDCentral",
    #         # "weight_mu2_LowPt_MuonID_SF_MediumIDCentral",
    #         "weight_mu2_MuonID_SF_MediumIDLoosePFIsoCentral",
    #         # "weight_mu2_MuonID_SF_LoosePFIsoCentral",
    #         "weight_mu2_MuonID_SF_MediumID_TrkCentral",
    #     ]
    # }

    ID_weights_dict = {
        # "muMu": [
        #     "weight_mu1_tightID",
        #     "weight_mu1_tightID_tightIso",
        #     "weight_mu2_tightID",
        #     "weight_mu2_tightID_tightIso",
        # ]
        "muMu": [
            "weight_mu1_mediumID",
            "weight_mu1_mediumID_looseIso",
            "weight_mu2_mediumID",
            "weight_mu2_mediumID_looseIso",
        ]
    }


    # should be moved to config
    weights_to_apply.extend(ID_weights_dict[channel])
    weights_to_apply.extend(trg_weights_dict[channel])

    total_weight = "*".join(weights_to_apply)
    # print(total_weight)
    return total_weight


class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    # def RescaleXS(self):
    #     import yaml
    #     xsFile = self.config["crossSectionsFile"]
    #     xsFilePath = os.path.join(os.environ["ANALYSIS_PATH"], xsFile)
    #     with open(xsFilePath, "r") as xs_file:
    #         xs_dict = yaml.safe_load(xs_file)
    #     xs_condition = f"DY" in self.config["process_name"] #== "DY"
    #     xs_to_scale = (
    #         xs_dict["DY_NNLO_QCD+NLO_EW"]["crossSec"] if xs_condition else "1.f"
    #     )
    #     weight_XS_string = f"xs_to_scale/current_xs" if xs_condition else "1."
    #     total_denunmerator_nJets = 5378.0 / 3 + 1017.0 / 3 + 385.5 / 3
    #     self.df = self.df.Define(f"current_xs", f"{total_denunmerator_nJets}")
    #     self.df = self.df.Define(f"xs_to_scale", f"{xs_to_scale}")
    #     self.df = self.df.Define(f"weight_XS", weight_XS_string)

    def defineTriggers(self):
        for ch in self.config["channelSelection"]:
            for trg in self.config["triggers"][ch]:
                trg_name = "HLT_" + trg
                self.colToSave.append(trg_name)
                if trg_name not in self.df.GetColumnNames():
                    print(f"{trg_name} not present in colNames")
                    self.df = self.df.Define(trg_name, "1")

    # def defineSampleType(self):
    #     self.df = self.df.Define(
    #         f"sample_type",
    #         f"""std::string process_name = "{self.config["process_name"]}"; return process_name;""",
    #     )

    def defineRegions(self):
        region_defs = self.config["MuMuMassRegions"]
        for reg_name, reg_cut in region_defs.items():
            self.df = self.df.Define(reg_name, reg_cut)
            self.colToSave.append(reg_name)

    def SignRegionDef(self):
        self.df = self.df.Define("OS", "mu1_charge*mu2_charge < 0")
        self.colToSave.append("OS")
        self.df = self.df.Define("SS", "!OS")
        self.colToSave.append("SS")

    def defineCategories(self):  # at the end
        singleMuTh = self.config["singleMu_th"][self.period]
        for category_to_def in self.config["category_definition"].keys():
            category_name = category_to_def
            cat_str = self.config["category_definition"][category_to_def].format(
                MuPtTh=singleMuTh
            )
            self.df = self.df.Define(category_to_def, cat_str)
            self.colToSave.append(category_to_def)

    def defineChannels(self):
        self.df = self.df.Define(f"muMu", f"return true;")
        self.colToSave.append("muMu")

    def __init__(
        self,
        df,
        config,
        period,
        isData=False,
        isCentral=True,
        wantTriggerSFErrors=False,
        colToSave=[],
    ):
        super(DataFrameBuilderForHistograms, self).__init__(df)
        self.config = config
        self.isData = isData
        self.period = period
        self.colToSave = colToSave
        self.wantTriggerSFErrors = wantTriggerSFErrors


def PrepareDfForHistograms(dfForHistograms):
    dfForHistograms.df = RescaleXS(dfForHistograms.df,dfForHistograms.config)
    dfForHistograms.defineChannels()
    dfForHistograms.defineTriggers()
    dfForHistograms.SignRegionDef()

    dfForHistograms.df = AddScaReOnBS(dfForHistograms.df, dfForHistograms.period, dfForHistograms.isData)
    dfForHistograms.df = AddRoccoR(dfForHistograms.df, dfForHistograms.period, dfForHistograms.isData)

    dfForHistograms.df = RedefineIsoTrgAndIDWeights(dfForHistograms.df, dfForHistograms.period) # here nano pT is needed in any case because corrections are derived on nano pT

    # if not dfForHistograms.isData:
    #     defineTriggerWeights(dfForHistograms)
    #     if dfForHistograms.wantTriggerSFErrors:
    #         defineTriggerWeightsErrors(dfForHistograms)

    dfForHistograms.df = AddNewDYWeights(dfForHistograms.df, dfForHistograms.period, f"DY" in dfForHistograms.config["process_name"]) # here nano pT is needed in any case because corrections are derived on nano pT

    dfForHistograms.df = GetAllMuMuPtRelatedObservables(dfForHistograms.df) # this can go before redefinition of pT because it defines for all the specific combinations

    dfForHistograms.df = GetMuMuMassResolution(dfForHistograms.df) # this can go before redefinition of pT because it defines for all the specific combinations
    dfForHistograms.df = JetCollectionDef(dfForHistograms.df)
    dfForHistograms.df = VBFJetSelection(dfForHistograms.df)
    dfForHistograms.df = RedefineMuonsPt(dfForHistograms.df, dfForHistograms.config["pt_to_use"])
    dfForHistograms.df = RedefineDiMuonObservables(dfForHistograms.df)

    dfForHistograms.df = VBFJetMuonsObservables(dfForHistograms.df) # from here, the pT is needed to be specified as it depends on which muon pT to choose.


    dfForHistograms.defineRegions() # this depends on which muon pT to choose.
    dfForHistograms.defineCategories() # this depends on which muon  pT to choose.
    return dfForHistograms


def PrepareDfForNNInputs(dfBuilder):
    # dfBuilder.RescaleXS()
    dfBuilder.defineChannels()
    dfBuilder.defineTriggers()
    dfBuilder.AddScaReOnBS()
    dfBuilder.df = GetMuMuObservables(dfBuilder.df)
    dfBuilder.df = GetMuMuMassResolution(dfBuilder.df)
    dfBuilder.df = JetCollectionDef(dfBuilder.df)
    # dfBuilder.df = VBFJetSelection(dfBuilder.df)
    # dfBuilder.df = VBFJetMuonsObservables(dfBuilder.df)
    # dfBuilder.df = GetSoftJets(dfBuilder.df)
    # dfBuilder.defineRegions()
    dfBuilder.defineCategories()
    return dfBuilder

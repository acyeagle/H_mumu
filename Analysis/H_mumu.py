import ROOT
import sys
import os

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.HistHelper import *
from FLAF.Common.Utilities import *
from Analysis.GetTriggerWeights import *
from FLAF.Common import Utilities

for header in [
    "FLAF/include/Utilities.h",
    "include/Helper.h",
    "include/HmumuCore.h",
    "FLAF/include/AnalysisTools.h",
    "FLAF/include/AnalysisMath.h",
]:
    DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)

JetObservables = [
    "PNetRegPtRawCorr",
    "PNetRegPtRawCorrNeutrino",
    "PNetRegPtRawRes",
    "UParTAK4RegPtRawCorr",
    "UParTAK4RegPtRawCorrNeutrino",
    "UParTAK4RegPtRawRes",
    "area",
    "btagDeepFlavB",
    "btagDeepFlavCvB",
    "btagDeepFlavCvL",
    "btagDeepFlavQG",
    "btagPNetB",
    "btagPNetCvB",
    "btagPNetCvL",
    "btagPNetCvNotB",
    "btagPNetQvG",
    "btagPNetTauVJet",
    "btagUParTAK4B",
    "btagUParTAK4CvB",
    "btagUParTAK4CvL",
    "btagUParTAK4CvNotB",
    "btagUParTAK4QvG",
    "btagUParTAK4TauVJet",
    "chEmEF",
    "chHEF",
    "chMultiplicity",
    "electronIdx1",
    "electronIdx2",
    "eta",
    "genJetIdx",
    "hfEmEF",
    "hfHEF",
    "hfadjacentEtaStripsSize",
    "hfcentralEtaStripSize",
    "hfsigmaEtaEta",
    "hfsigmaPhiPhi",
    "jetId",
    "mass",
    "muEF",
    "muonIdx1",
    "muonIdx2",
    "muonSubtrFactor",
    "nConstituents",
    "nElectrons",
    "nMuons",
    "nSVs",
    "neEmEF",
    "neHEF",
    "neMultiplicity",
    "phi",
    "pt",
    "puIdDisc",
    "puId_beta",
    "puId_dR2Mean",
    "puId_frac01",
    "puId_frac02",
    "puId_frac03",
    "puId_frac04",
    "puId_jetR",
    "puId_jetRchg",
    "puId_majW",
    "puId_minW",
    "puId_nCharged",
    "puId_ptD",
    "puId_pull",
    "rawFactor",
    "svIdx1",
    "svIdx2",
    "ptRes",
    "vetoMap",
    # "passJetIdTight",
    # "passJetIdTightLepVeto",
    # "isInsideVetoRegion"
]
JetObservablesMC = ["hadronFlavour", "partonFlavour", "genJetIdx"]


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


def JetCollectionDef(df):
    if "Jet_idx" not in df.GetColumnNames():
        print("Jet_idx not in df.GetColumnNames")
        df = df.Define(f"Jet_idx", f"CreateIndexes(Jet_pt.size())")
    df = df.Define(
        f"Jet_p4",
        f"GetP4(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_idx)",
    )

    #### Jet PreSelection ####
    df = df.Define(
        "Jet_preSel",
        f"""v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4))< 4.7 && (Jet_jetId & 2) """,
    )
    df = df.Define(
        "Jet_preSel_andDeadZoneVetoMap",
        "Jet_preSel && !Jet_vetoMap",
    )

    df = df.Define(
        f"Jet_NoOverlapWithMuons",
        f"RemoveOverlaps(Jet_p4, Jet_preSel_andDeadZoneVetoMap, {{mu1_p4, mu2_p4}}, 0.4)",
    )
    df = df.Define(
        f"SelectedJet_p4",
        f"Jet_p4[Jet_NoOverlapWithMuons]",
    )
    df = df.Define(
        f"SelectedJet_index",
        f"Jet_idx[Jet_NoOverlapWithMuons]",
    )

    ### Final state definitions: removing bTagged jets - deepJet ####
    df = df.Define(
        "Jet_btag_Veto_loose_deepJet",
        "Jet_btagDeepFlavB >= 0.0614 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )
    df = df.Define(
        "Jet_btag_Veto_medium_deepJet",
        "Jet_btagDeepFlavB >= 0.3196 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )
    df = df.Define(
        "JetTagSel_deepJet",
        "Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_medium_deepJet].size() < 1  && Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_loose_deepJet].size() < 2",
    )

    #### Final state definitions: removing bTagged jets - pNet ####
    df = df.Define(
        "Jet_btag_Veto_loose",
        "Jet_btagPNetB >= 0.0499 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )  # 0.0499 is the loose working point for PNet B-tagging in Run3
    df = df.Define(
        "Jet_btag_Veto_medium",
        "Jet_btagPNetB >= 0.2605 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )  # 0.2605 is the medium working point for PNet B-tagging in Run3
    # df = df.Define("Jet_Veto_tight", "Jet_btagPNetB >= 0.6484")  # 0.6484 is the tight working point for PNet B-tagging in Run3
    df = df.Define(
        "JetTagSel",
        "Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_medium].size() < 1  && Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_loose].size() < 2",
    )
    return df


def VBFNetJetCollectionDef(df):
    if "Jet_idx" not in df.GetColumnNames():
        print("Jet_idx not in df.GetColumnNames")
        df = df.Define(f"Jet_idx", f"CreateIndexes(Jet_pt.size())")
    df = df.Define(
        f"Jet_p4",
        f"GetP4(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_idx)",
    )

    #### Jet PreSelection ####
    df = df.Define(
        "Jet_preSel",
        f"""v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4))< 4.7 && (Jet_jetId & 2) """,
    )
    df = df.Define(
        "Jet_preSel_andDeadZoneVetoMap",
        "Jet_preSel && !Jet_vetoMap",
    )

    df = df.Define(
        f"Jet_NoOverlapWithMuons",
        f"RemoveOverlaps(Jet_p4, Jet_preSel_andDeadZoneVetoMap, {{mu1_p4, mu2_p4}}, 0.4)",
    )
    df = df.Define(
        f"SelectedJet_p4",
        f"Jet_p4[Jet_NoOverlapWithMuons]",
    )
    df = df.Define(
        f"SelectedJet_index",
        f"Jet_idx[Jet_NoOverlapWithMuons]",
    )

    ### Final state definitions: removing bTagged jets - deepJet ####
    df = df.Define(
        "Jet_btag_Veto_loose_deepJet",
        "Jet_btagDeepFlavB >= 0.0614 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )
    df = df.Define(
        "Jet_btag_Veto_medium_deepJet",
        "Jet_btagDeepFlavB >= 0.3196 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )
    df = df.Define(
        "JetTagSel_deepJet",
        "Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_medium_deepJet].size() < 1  && Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_loose_deepJet].size() < 2",
    )

    #### Final state definitions: removing bTagged jets - pNet ####
    df = df.Define(
        "Jet_btag_Veto_loose",
        "Jet_btagPNetB >= 0.0499 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )  # 0.0499 is the loose working point for PNet B-tagging in Run3
    df = df.Define(
        "Jet_btag_Veto_medium",
        "Jet_btagPNetB >= 0.2605 && abs(v_ops::eta(Jet_p4))< 2.5 ",
    )  # 0.2605 is the medium working point for PNet B-tagging in Run3
    # df = df.Define("Jet_Veto_tight", "Jet_btagPNetB >= 0.6484")  # 0.6484 is the tight working point for PNet B-tagging in Run3
    df = df.Define(
        "JetTagSel",
        "Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_medium].size() < 1  && Jet_p4[Jet_NoOverlapWithMuons && Jet_btag_Veto_loose].size() < 2",
    )

    df = df.Define(
        "VBFCandJet_selection", 
        "Jet_NoOverlapWithMuons && Jet_pt > 25 && ((ROOT::VecOps::abs(Jet_eta) < 2.5) || (ROOT::VecOps::abs(Jet_eta) > 3.0) || (Jet_pt > 50));"
    )

    # Add the desired variables

    jet_vars = ['pt', 'eta', 'phi', 'btagPNetB', 'btagPNetCvB', 'btagPNetCvL', 'btagPNetCvNotB', 'btagPNetQvG', 'btagPNetTauVJet', 'puIdDisc']

    for var in jet_vars:
        df = df.Define(
            f"FilteredJet_{var}", f"Jet_{var}[VBFCandJet_selection]"
        )

    # df = df.Define(
    #     "FilteredJet_pt", "Jet_pt[VBFCandJet_selection]"
    # )
    # df = df.Define(
    #     "FilteredJet_eta", "Jet_eta[VBFCandJet_selection]"
    # )
    # df = df.Define(
    #     "FilteredJet_phi", "Jet_phi[VBFCandJet_selection]"
    # )
    # df = df.Define(
    #     "FilteredJet_btagPNetQvG", "Jet_btagPNetQvG[VBFCandJet_selection]"
    # )
    # df = df.Define(
    #     "FilteredJet_puIdDisc", "Jet_puIdDisc[VBFCandJet_selection]"
    # )

    # and do the same for whatever other variables
    # Jet_btagPNetQvG 
    # Jet_puIdDisc

    return df


def VBFJetSelection(df):
    df = df.Define("VBFJetCand", "FindVBFJets(Jet_p4,Jet_NoOverlapWithMuons)")
    df = df.Define("HasVBF", "return static_cast<bool>(VBFJetCand.isVBF) ")

    df = df.Define(
        "m_jj",
        "if (HasVBF) return static_cast<float>(VBFJetCand.m_inv); return -1000.f",
    )
    df = df.Define(
        "delta_eta_jj",
        "if (HasVBF) return static_cast<float>(VBFJetCand.eta_separation); return -1000.f",
    )
    df = df.Define(
        "j1_idx",
        "if (HasVBF) return static_cast<int>(VBFJetCand.leg_index[0]); return -1000; ",
    )
    df = df.Define(
        "j2_idx",
        "if (HasVBF) return static_cast<int>(VBFJetCand.leg_index[1]); return -1000; ",
    )
    df = df.Define(
        "j1_pt",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Pt()); return -1000.f; ",
    )
    df = df.Define(
        "j2_pt",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Pt()); return -1000.f; ",
    )
    df = df.Define(
        "j1_eta",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Eta()); return -1000.f; ",
    )
    df = df.Define(
        "j2_eta",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Eta()); return -1000.f; ",
    )
    df = df.Define(
        "j1_phi",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Phi()); return -1000.f; ",
    )
    df = df.Define(
        "j2_phi",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Phi()); return -1000.f; ",
    )
    df = df.Define(
        "j1_y",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[0].Rapidity()); return -1000.f; ",
    )
    df = df.Define(
        "j2_y",
        "if (HasVBF) return static_cast<float>(VBFJetCand.leg_p4[1].Rapidity()); return -1000.f; ",
    )
    df = df.Define(
        "delta_phi_jj",
        "if (HasVBF) return static_cast<float>(ROOT::Math::VectorUtil::DeltaPhi( VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1] ) ); return -1000.f;",
    )

    df = df.Define(f"pt_jj", "(VBFJetCand.leg_p4[0]+VBFJetCand.leg_p4[1]).Pt()")
    df = df.Define(
        "VBFjets_pt",
        f"RVecF void_pt {{}} ; if (HasVBF) return v_ops::pt(VBFJetCand.legs_p4); return void_pt;",
    )
    df = df.Define(
        "VBFjets_eta",
        f"RVecF void_eta {{}} ; if (HasVBF) return v_ops::eta(VBFJetCand.legs_p4); return void_eta;",
    )
    df = df.Define(
        "VBFjets_phi",
        f"RVecF void_phi {{}} ; if (HasVBF) return v_ops::phi(VBFJetCand.legs_p4); return void_phi;",
    )
    df = df.Define(
        "VBFjets_y",
        f"RVecF void_y {{}} ; if (HasVBF) return v_ops::rapidity(VBFJetCand.legs_p4); return void_y;",
    )
    for var in JetObservables:
        if f"Jet_{var}" not in df.GetColumnNames():
            continue
        if f"j1_{var}" not in df.GetColumnNames():
            df = df.Define(
                "j1_" + var,
                f"if (HasVBF && j1_idx >= 0) return static_cast<float>(Jet_{var}[j1_idx]); return -1000.f;",
            )
        if f"j2_{var}" not in df.GetColumnNames():
            df = df.Define(
                "j2_" + var,
                f"if (HasVBF && j2_idx >= 0) return static_cast<float>(Jet_{var}[j2_idx]); return -1000.f;",
            )

    return df


def GetMuMuObservables(df):
    for idx in [0, 1]:
        df = Utilities.defineP4(df, f"mu{idx+1}")
        df = df.Define(
            f"mu{idx+1}_p4_BS",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_bsConstrainedPt,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
        df = df.Define(
            f"mu{idx+1}_p4_nano",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt_nano,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
        df = df.Define(
            f"mu{idx+1}_p4_BS_ScaRe",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_BS_pt_1_corr,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
    df = df.Define(f"pt_mumu", "(mu1_p4+mu2_p4).Pt()")
    df = df.Define(f"pt_mumu_nano", "(mu1_p4_nano+mu2_p4_nano).Pt()")
    df = df.Define(f"pt_mumu_BS", "(mu1_p4_BS+mu2_p4_BS).Pt()")
    df = df.Define(f"pt_mumu_BS_ScaRe", "(mu1_p4_BS_ScaRe+mu2_p4_BS_ScaRe).Pt()")
    df = df.Define(f"y_mumu", "(mu1_p4+mu2_p4).Rapidity()")
    df = df.Define(f"eta_mumu", "(mu1_p4+mu2_p4).Eta()")
    df = df.Define(f"phi_mumu", "(mu1_p4+mu2_p4).Phi()")
    df = df.Define("m_mumu", "static_cast<float>((mu1_p4+mu2_p4).M())")
    df = df.Define("m_mumu_nano", "static_cast<float>((mu1_p4_nano+mu2_p4_nano).M())")
    df = df.Define("m_mumu_BS", "static_cast<float>((mu1_p4_BS+mu2_p4_BS).M())")
    df = df.Define(
        "m_mumu_BS_ScaRe", "static_cast<float>((mu1_p4_BS_ScaRe+mu2_p4_BS_ScaRe).M())"
    )
    for idx in [0, 1]:
        df = df.Define(f"mu{idx+1}_pt_rel", f"mu{idx+1}_pt/m_mumu")
        df = df.Define(f"mu{idx+1}_pt_rel_BS", f"mu{idx+1}_bsConstrainedPt/m_mumu_BS")
        df = df.Define(f"mu{idx+1}_pt_rel_nano", f"mu{idx+1}_pt_nano/m_mumu_nano")
        df = df.Define(
            f"mu{idx+1}_pt_rel_BS_ScaRe", f"mu{idx+1}_BS_pt_1_corr/m_mumu_BS_ScaRe"
        )

    df = df.Define("dR_mumu", "ROOT::Math::VectorUtil::DeltaR(mu1_p4, mu2_p4)")

    df = df.Define("Ebeam", "13600.0/2")
    df = df.Define("cosTheta_Phi_CS", "ComputeCosThetaPhiCS(mu1_p4, mu2_p4,  Ebeam)")
    df = df.Define("cosTheta_CS", "static_cast<float>(std::get<0>(cosTheta_Phi_CS))")
    df = df.Define("phi_CS", "static_cast<float>(std::get<1>(cosTheta_Phi_CS))")
    return df


def GetMuMuMassResolution(df):
    delta_mu_expr = "sqrt( 0.5 * (pow( ({0}/{1}), 2) + pow( ({2}/{3}), 2) ) ) "
    df = df.Define(
        "m_mumu_resolution",
        delta_mu_expr.format(
            "mu1_pt",
            "(mu1_pt-mu1_pt_nano)/mu1_pt",
            "mu2_pt",
            "(mu2_pt-mu2_pt_nano)/mu2_pt",
        ),
    )
    df = df.Define(
        "m_mumu_resolution_relerr",
        delta_mu_expr.format(
            "mu1_pt", "(mu1_pt-mu1_pt_nano)", "mu2_pt", "(mu2_pt-mu2_pt_nano)"
        ),
    )
    df = df.Define(
        "m_mumu_resolution_bsConstrained",
        delta_mu_expr.format(
            "mu1_bsConstrainedPt",
            "mu1_bsConstrainedPtErr",
            "mu2_bsConstrainedPt",
            "mu2_bsConstrainedPtErr",
        ),
    )
    return df


def VBFJetMuonsObservables(df):
    df = df.Define(
        "Zepperfield_Var",
        "if (HasVBF) return static_cast<float>((y_mumu - 0.5*(j1_y+j2_y))/std::abs(j1_y - j2_y)); return -10000.f;",
    )
    df = df.Define(
        "pT_all_sum",
        "if(HasVBF) return static_cast<float>(pT_sum ({mu1_p4, mu2_p4, VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1]})); return -10000.f;",
    )
    df = df.Define(
        "R_pt",
        "if(HasVBF) return static_cast<float>((pT_all_sum)/(pt_mumu + j1_pt + j2_pt)); return -10000.f;",
    )
    df = df.Define(
        "pT_jj_sum",
        "if(HasVBF) return static_cast<float>(pT_sum ({VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1]})); return -10000.f;",
    )
    df = df.Define(
        "pt_centrality",
        "if(HasVBF) return static_cast<float>(( (pt_mumu-0.5*(pT_jj_sum)) / pT_diff(VBFJetCand.leg_p4[0], VBFJetCand.leg_p4[1]) )); return -10000.f;",
    )

    df = df.Define(
        "minDeltaPhi",
        "if(HasVBF) return static_cast<float>(std::min(ROOT::Math::VectorUtil::DeltaPhi( (mu1_p4+mu2_p4), VBFJetCand.leg_p4[0]), ROOT::Math::VectorUtil::DeltaPhi((mu1_p4+mu2_p4), VBFJetCand.leg_p4[1]) ) )  ; return -10000.f;",
    )
    df = df.Define(
        "minDeltaEta",
        "if(HasVBF) return static_cast<float>(std::min(std::abs(eta_mumu - j1_eta),std::abs(eta_mumu - j2_eta))) ; return -10000.f;",
    )
    df = df.Define(
        "minDeltaEtaSigned",
        "if(HasVBF) return static_cast<float>(std::min((eta_mumu - j1_eta),(eta_mumu - j2_eta))) ; return -10000.f;",
    )

    return df


def GetSoftJets(df):
    df = df.Define(
        "SoftJet_def_vtx", "(Jet_svIdx1 < 0 && Jet_svIdx2< 0 ) "
    )  # no secondary vertex associated
    df = df.Define("SoftJet_def_pt", " (Jet_pt>2) ")  # pT > 2 GeV
    df = df.Define(
        "SoftJet_def_muon",
        "(Jet_idx != mu1_jetIdx && Jet_idx != mu2_jetIdx)",
    )  # TMP PATCH. For next round it will be changed to the commented one in next line --> the muon index of the jets (because there can be muons associated to jets) has to be different than the signal muons (i.e. those coming from H decay)
    # df = df.Define(
    #     "SoftJet_def_muon",
    #     "(Jet_muonIdx1 != mu1_index && Jet_muonIdx2 != mu2_index && Jet_muonIdx2 != mu1_index && Jet_muonIdx2 != mu2_index)",
    # )  # mu1_idx and mu2_idx are not present in the current anaTuples, but need to be introduced for next round . The idx is the index in the original muon collection as well as Jet_muonIdx()

    df = df.Define(
        "SoftJet_def_VBF",
        " (HasVBF && Jet_idx != j1_idx && Jet_idx != j2_idx) ",
    )  # if it is a VBF event, the soft jets are not the VBF jets
    df = df.Define("SoftJet_def_noVBF", " (!(HasVBF)) ")

    df = df.Define(
        "SoftJet_def",
        "SoftJet_def_vtx && SoftJet_def_pt && SoftJet_def_muon && (SoftJet_def_VBF || SoftJet_def_noVBF )",
    )

    df = df.Define("N_softJet", "Jet_p4[SoftJet_def].size()")
    df = df.Define("SoftJet_energy", "v_ops::energy(Jet_p4[SoftJet_def])")
    df = df.Define("SoftJet_Et", "v_ops::Et(Jet_p4[SoftJet_def])")
    df = df.Define("SoftJet_HtCh_fraction", "Jet_chHEF[SoftJet_def]")
    df = df.Define("SoftJet_HtNe_fraction", "Jet_neHEF[SoftJet_def]")
    if "Jet_hfHEF" in df.GetColumnNames():
        df = df.Define("SoftJet_HtHF_fraction", "Jet_hfHEF[SoftJet_def]")
    for var in JetObservables:
        if f"SoftJet_{var}" not in df.GetColumnNames():
            if (
                f"SoftJet_{var}" not in df.GetColumnNames()
                and f"Jet_{var}" in df.GetColumnNames()
            ):
                df = df.Define(f"SoftJet_{var}", f"Jet_{var}[SoftJet_def]")
    for var in JetObservablesMC:
        if (
            f"SoftJet_{var}" not in df.GetColumnNames()
            and f"Jet_{var}" in df.GetColumnNames()
        ):
            df = df.Define(f"SoftJet_{var}", f"Jet_{var}[SoftJet_def]")
    return df


def defineP4AndInvMass(df):
    if "Jet_idx" not in df.GetColumnNames():
        print("Jet_idx not in df.GetColumnNames")
        df = df.Define(f"Jet_idx", f"CreateIndexes(Jet_pt.size())")
    df = df.Define(
        f"Jet_p4",
        f"GetP4(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_idx)",
    )
    for idx in [0, 1]:
        df = Utilities.defineP4(df, f"mu{idx+1}")
    df = df.Define(f"pt_ll", "(mu1_p4+mu2_p4).Pt()")

    df = df.Define("m_mumu", "static_cast<float>((mu1_p4+mu2_p4).M())")
    df = df.Define("dR_mumu", "ROOT::Math::VectorUtil::DeltaR(mu1_p4, mu2_p4)")

    return df


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
        "weight_EWKCorr_VptCentral",
        "weight_DYw_DYWeightCentral",
    ]  # ,"weight_EWKCorr_ewcorrCentral"] #

    trg_weights_dict = {
        "muMu": ["weight_trigSF_singleMu"]
    }  # ["weight_mu1_TrgSF_singleMu_Central", "weight_mu2_TrgSF_singleMu_Central"]}  # ["weight_trigSF_singleMu"],
    ID_weights_dict = {
        "muMu": [
            "weight_mu1_HighPt_MuonID_SF_MediumIDCentral",
            "weight_mu1_HighPt_MuonID_SF_RecoCentral",
            "weight_mu1_LowPt_MuonID_SF_MediumIDCentral",
            "weight_mu1_MuonID_SF_LoosePFIsoCentral",
            "weight_mu1_MuonID_SF_MediumIDLoosePFIsoCentral",
            "weight_mu1_MuonID_SF_MediumID_TrkCentral",
            "weight_mu2_HighPt_MuonID_SF_MediumIDCentral",
            "weight_mu2_HighPt_MuonID_SF_RecoCentral",
            "weight_mu2_LowPt_MuonID_SF_MediumIDCentral",
            "weight_mu2_MuonID_SF_LoosePFIsoCentral",
            "weight_mu2_MuonID_SF_MediumIDLoosePFIsoCentral",
            "weight_mu2_MuonID_SF_MediumID_TrkCentral",
        ]
    }
    # should be moved to config
    # what about :
    #   weight_mu1_HighPt_MuonID_SF_MediumIDLooseRelIsoHLTCentral ?
    #   weight_mu2_HighPt_MuonID_SF_MediumIDLooseRelIsoHLTCentral
    #   weight_mu1_HighPt_MuonID_SF_MediumIdLooseRelTkIsoCentral ?
    #   weight_mu2_HighPt_MuonID_SF_MediumIdLooseRelTkIsoCentral ?
    weights_to_apply.extend(ID_weights_dict[channel])
    weights_to_apply.extend(trg_weights_dict[channel])

    total_weight = "*".join(weights_to_apply)
    # print(total_weight)
    return total_weight


class DataFrameBuilderForHistograms(DataFrameBuilderBase):

    def RescaleXS(self):
        import yaml

        xsFile = self.config["crossSectionsFile"]
        xsFilePath = os.path.join(os.environ["ANALYSIS_PATH"], xsFile)
        with open(xsFilePath, "r") as xs_file:
            xs_dict = yaml.safe_load(xs_file)
        xs_condition = self.config["process_name"] == "DY"
        xs_to_scale = (
            xs_dict["DY_NNLO_QCD+NLO_EW"]["crossSec"] if xs_condition else "1.f"
        )
        weight_XS_string = f"xs_to_scale/current_xs" if xs_condition else "1."
        total_denunmerator_nJets = 5378.0 / 3 + 1017.0 / 3 + 385.5 / 3
        self.df = self.df.Define(f"current_xs", f"{total_denunmerator_nJets}")
        self.df = self.df.Define(f"xs_to_scale", f"{xs_to_scale}")
        self.df = self.df.Define(f"weight_XS", weight_XS_string)

    def defineTriggers(self):
        for ch in self.config["channelSelection"]:
            for trg in self.config["triggers"][ch]:
                trg_name = "HLT_" + trg
                self.colToSave.append(trg_name)
                if trg_name not in self.df.GetColumnNames():
                    print(f"{trg_name} not present in colNames")
                    self.df = self.df.Define(trg_name, "1")

    def defineSampleType(self):
        self.df = self.df.Define(
            f"sample_type",
            f"""std::string process_name = "{self.config["process_name"]}"; return process_name;""",
        )

    def AddScaReOnBS(self):
        import correctionlib

        period_files = {
            "Run3_2022": "2022_Summer22",
            "Run3_2022EE": "2022_Summer22EE",
            "Run3_2023": "2023_Summer23",
            "Run3_2023BPix": "2023_Summer23BPix",
        }
        correctionlib.register_pyroot_binding()
        file_name = period_files.get(self.period, "")
        analysis_path = os.environ["ANALYSIS_PATH"]
        ROOT.gROOT.ProcessLine(
            f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Corrections/data/MUO/MuonScaRe/{file_name}.json");'
        )
        ROOT.gROOT.ProcessLine(f'#include "{analysis_path}/include/MuonScaRe.cc"')
        for mu_idx in [1, 2]:
            if self.isData:
                # Data apply scale correction
                self.df = self.df.Define(
                    f"mu{mu_idx}_BS_pt_1_corr",
                    f"pt_scale(1, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
                )
            else:
                self.df = self.df.Define(
                    f"mu{mu_idx}_BS_pt_1_scale_corr",
                    f"pt_scale(0, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
                )

                self.df = self.df.Define(
                    f"mu{mu_idx}_BS_pt_1_corr",
                    f"pt_resol(mu{mu_idx}_BS_pt_1_scale_corr, mu{mu_idx}_eta, float(mu{mu_idx}_nTrackerLayers))",
                )
                # # MC evaluate scale uncertainty
                # df_mc = df_mc.Define(
                #     'pt_1_scale_corr_up',
                #     'pt_scale_var(pt_1_corr, eta_1, phi_1, charge_1, "up")'
                # )
                # df_mc = df_mc.Define(
                #     'pt_1_scale_corr_dn',
                #     'pt_scale_var(pt_1_corr, eta_1, phi_1, charge_1, "dn")'
                # )

                # # MC evaluate resolution uncertainty
                # df_mc = df_mc.Define(
                #     "pt_1_corr_resolup",
                #     'pt_resol_var(pt_1_scale_corr, pt_1_corr, eta_1, "up")'
                # )
                # df_mc = df_mc.Define(
                #     "pt_1_corr_resoldn",
                #     'pt_resol_var(pt_1_scale_corr, pt_1_corr, eta_1, "dn")'
                # )

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
    dfForHistograms.RescaleXS()
    dfForHistograms.defineChannels()
    # dfForHistograms.defineSampleType()
    dfForHistograms.defineTriggers()
    dfForHistograms.AddScaReOnBS()
    dfForHistograms.df = GetMuMuObservables(dfForHistograms.df)
    dfForHistograms.df = GetMuMuMassResolution(dfForHistograms.df)
    dfForHistograms.df = JetCollectionDef(dfForHistograms.df)
    dfForHistograms.df = VBFJetSelection(dfForHistograms.df)
    dfForHistograms.df = VBFJetMuonsObservables(dfForHistograms.df)
    dfForHistograms.df = GetSoftJets(dfForHistograms.df)
    if not dfForHistograms.isData:
        defineTriggerWeights(dfForHistograms)
    #     if dfForHistograms.wantTriggerSFErrors:
    #         defineTriggerWeightsErrors(dfForHistograms)
    dfForHistograms.SignRegionDef()
    dfForHistograms.defineRegions()
    dfForHistograms.defineCategories()
    return dfForHistograms


def PrepareDfForNNInputs(dfBuilder):
    #dfBuilder.RescaleXS()
    dfBuilder.defineChannels()
    dfBuilder.defineTriggers()
    dfBuilder.AddScaReOnBS()
    dfBuilder.df = GetMuMuObservables(dfBuilder.df)
    dfBuilder.df = GetMuMuMassResolution(dfBuilder.df)
    dfBuilder.df = JetCollectionDef(dfBuilder.df)
    dfBuilder.df = VBFJetSelection(dfBuilder.df)
    dfBuilder.df = VBFJetMuonsObservables(dfBuilder.df)
    dfBuilder.df = GetSoftJets(dfBuilder.df)
    dfBuilder.SignRegionDef()
    dfBuilder.defineRegions()
    dfBuilder.defineCategories()
    return dfBuilder

def PrepareDfForVBFNetworkInputs(dfBuilder):
    #dfBuilder.RescaleXS()
    dfBuilder.defineChannels()
    dfBuilder.defineTriggers()
    dfBuilder.AddScaReOnBS()
    dfBuilder.df = GetMuMuObservables(dfBuilder.df)
    dfBuilder.df = GetMuMuMassResolution(dfBuilder.df)
    dfBuilder.df = VBFNetJetCollectionDef(dfBuilder.df)
    dfBuilder.df = VBFJetSelection(dfBuilder.df)
    dfBuilder.df = VBFJetMuonsObservables(dfBuilder.df)
    dfBuilder.SignRegionDef()
    dfBuilder.defineRegions()
    dfBuilder.defineCategories()
    return dfBuilder

import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.Utilities import *
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
    "passJetIdTight",
    "passJetIdTightLepVeto",
    "isInsideVetoRegion"
]

JetObservablesMC = ["hadronFlavour", "partonFlavour", "genJetIdx"]


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
        f"""v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4))< 4.7 && (Jet_passJetIdTight) """,
    )
    # ed on “loose” selection: pT > 15 GeV and |η|<4.7 and passTightLepVetoId and (chEmEF + neEmEF) < 0.9)
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
    

    # df = df.Define(f"leadingjet_pt", "if (SelectedJet_p4.size()>0) return static_cast<float>(v_ops::pt(SelectedJet_p4)[0]); else return -1000.f;")
    # df = df.Define(f"leadingjet_eta", "if (SelectedJet_p4.size()>0) return static_cast<float>(v_ops::eta(SelectedJet_p4)[0]); else return -1000.f;")
    # df = df.Define(
    #     "VBFCandJet_selection", "Jet_NoOverlapWithMuons && Jet_pt > 25 && ((ROOT::VecOps::abs(Jet_eta) < 2.5 ||  )) ];
    # )
    # df = df.Define("VBFCand_pt", "Jet_pt[VBFCandJet_selection]")
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


def VBFNetJetCollectionDef(df):
    if "Jet_idx" not in df.GetColumnNames():
        print("Jet_idx not in df.GetColumnNames")
        df = df.Define(f"Jet_idx", f"CreateIndexes(Jet_pt.size())")


    if "Jet_p4" not in df.GetColumnNames():
        df = df.Define(
            f"Jet_p4",
            f"GetP4(Jet_pt, Jet_eta, Jet_phi, Jet_mass, Jet_idx)",
        )

    #### Jet PreSelection ####
    if "Jet_preSel" not in df.GetColumnNames():
        df = df.Define(
            "Jet_preSel",
            f"""v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4))< 4.7 && (Jet_passJetIdTight & 2) """,
        )

    if "Jet_preSel_andDeadZoneVetoMap" not in df.GetColumnNames():
        df = df.Define(
            "Jet_preSel_andDeadZoneVetoMap",
            "Jet_preSel && !Jet_vetoMap",
        )

    if "Jet_NoOverlapWithMuons" not in df.GetColumnNames():
        df = df.Define(
            f"Jet_NoOverlapWithMuons",
            f"RemoveOverlaps(Jet_p4, Jet_preSel_andDeadZoneVetoMap, {{mu1_p4, mu2_p4}}, 0.4)",
        )

    # Define jets for VBF selector network

    df = df.Define(
        "VBFCandJet_selection", 
        "Jet_NoOverlapWithMuons && Jet_pt > 20 && ((ROOT::VecOps::abs(Jet_eta) < 2.5) || (ROOT::VecOps::abs(Jet_eta) > 3.0) || (Jet_pt > 50));"
    )

    # Add the desired variables

    jet_vars = ['pt', 'eta', 'phi', 'btagPNetB', 'btagPNetCvB', 'btagPNetCvL', 'btagPNetCvNotB', 'btagPNetQvG', 'btagPNetTauVJet', 'puIdDisc']

    for var in jet_vars:
        df = df.Define(
            f"FilteredJet_{var}", f"Jet_{var}[VBFCandJet_selection]"
        )

    return df
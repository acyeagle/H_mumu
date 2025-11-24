import AnaProd.baseline as AnaBaseline
import FLAF.Common.BaselineSelection as CommonBaseline
from Corrections.Corrections import Corrections
from FLAF.Common.Utilities import *

lepton_legs = ["mu1", "mu2"]
offline_legs = ["mu1", "mu2"]
loadTF = False
loadHHBtag = False


Muon_observables = [
    "IP_cov00",
    "IP_cov10",
    "IP_cov11",
    "IP_cov20",
    "IP_cov21",
    "IP_cov22",
    "IPx",
    "IPy",
    "IPz",
    "bField_z",
    "bsConstrainedChi2",
    "bsConstrainedPt",
    "bsConstrainedPtErr",
    "charge",
    "dxy",
    "dxyErr",
    "dxybs",
    "dz",
    "dzErr",
    "eta",
    "fsrPhotonIdx",
    "genPartFlav",
    "genPartIdx",
    "genMatchIdx",
    "highPtId",
    "highPurity",
    "inTimeMuon",
    "ip3d",
    "ipLengthSig",
    "isGlobal",
    "isPFcand",
    "isStandalone",
    "isTracker",
    "jetIdx",
    "jetNDauCharged",
    "jetPtRelv2",
    "jetRelIso",
    "looseId",
    "mass",
    "mediumId",
    "mediumPromptId",
    "miniIsoId",
    "miniPFRelIso_all",
    "miniPFRelIso_chg",
    "multiIsoId",
    "mvaLowPt",
    "mvaMuID",
    "mvaMuID_WP",
    "nStations",
    "nTrackerLayers",
    "pdgId",
    "pfIsoId",
    "pfRelIso03_all",
    "pfRelIso03_chg",
    "pfRelIso04_all",
    "phi",
    "promptMVA",
    "pt",
    "ptErr",
    "puppiIsoId",
    "segmentComp",
    "sip3d",
    "softId",
    "softMva",
    "softMvaId",
    "softMvaRun3",
    "svIdx",
    "tightCharge",
    "tightId",
    "tkIsoId",
    "tkRelIso",
    "track_cov00",
    "track_cov10",
    "track_cov11",
    "track_cov20",
    "track_cov21",
    "track_cov22",
    "track_cov30",
    "track_cov31",
    "track_cov32",
    "track_cov33",
    "track_cov40",
    "track_cov41",
    "track_cov42",
    "track_cov43",
    "track_cov44",
    "track_dsz",
    "track_dxy",
    "track_lambda",
    "track_phi",
    "track_qoverp",
    "triggerIdLoose",
    "tunepRelPt",
]

PrimaryVertexObservables = [
    "PVBS_chi2",
    "PVBS_cov00",
    "PVBS_cov10",
    "PVBS_cov11",
    "PVBS_cov20",
    "PVBS_cov21",
    "PVBS_cov22",
    "PVBS_x",
    "PVBS_y",
    "PVBS_z",
    "PV_score",
    "PV_z",
    "PV_chi2",
    "PV_ndof",
    "PV_npvs",
    "PV_npvsGood",
    "PV_score",
    "PV_sumpt2",
    "PV_sumpx",
    "PV_sumpy",
    "PV_x",
    "PV_y",
    "PV_z",
]

SecondaryVertexObservables = [
    "SV_jetIdx",
    "SV_sVIdx",
    "SV_charge",
    "SV_chi2",
    "SV_dlen",
    "SV_dlenSig",
    "SV_dxy",
    "SV_dxySig",
    "SV_eta",
    "SV_mass",
    "SV_ndof",
    "SV_ntracks",
    "SV_pAngle",
    "SV_phi",
    "SV_pt",
    "SV_x",
    "SV_y",
    "SV_z",
]

PUObservables = [
    "Pileup_gpudensity",
    "Pileup_nPU",
    "Pileup_pthatmax",
    "Pileup_pudensity",
    "Pileup_sumEOOT",
    "Pileup_sumLOOT",
]

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
    "vetoMapEle",
    "passJetIdTight",
    "passJetIdTightLepVeto",
    "isInsideVetoRegion"
]

JetObservablesMC = ["hadronFlavour", "partonFlavour"]

FSRPhotonObservables = [
    "FsrPhoton_dROverEt2",
    "FsrPhoton_electronIdx",
    "FsrPhoton_eta",
    "FsrPhoton_muonIdx",
    "FsrPhoton_phi",
    "FsrPhoton_pt",
    "FsrPhoton_relIso03",
]

SoftActivityJetObservables = [
    "SoftActivityJet_eta",
    "SoftActivityJet_phi",
    "SoftActivityJet_pt",
    "SoftActivityJetHT",
    "SoftActivityJetHT10",
    "SoftActivityJetHT2",
    "SoftActivityJetHT5",
    "SoftActivityJetNjets10",
    "SoftActivityJetNjets2",
    "SoftActivityJetNjets5",
]


defaultColToSave = [
    "FullEventId",
    "luminosityBlock",
    "run",
    "event",
    "sample_type",
    "period",
    "isData",
    "PV_npvs",
    "BeamSpot_sigmaZ",
    "BeamSpot_sigmaZError",
    "BeamSpot_type",
    "BeamSpot_z",
    "BeamSpot_zError",
]

additional_VBFStudies_vars = [
    "GenJet_eta",
    "GenJet_hadronFlavour",
    "GenJet_mass",
    "GenJet_nBHadrons",
    "GenJet_nCHadrons",
    "GenJet_partonFlavour",
    "GenJet_phi",
    "GenJet_pt",
    "LHEPart_eta",
    "LHEPart_incomingpz",
    "LHEPart_mass",
    "LHEPart_pdgId",
    "LHEPart_phi",
    "LHEPart_pt",
    "LHEPart_spin",
    "LHEPart_status",
    "GenPart_eta",
    "GenPart_genPartIdxMother",
    "GenPart_iso",
    "GenPart_mass",
    "GenPart_pdgId",
    "GenPart_phi",
    "GenPart_pt",
    "GenPart_status",
    "GenPart_statusFlags",
    "GenPart_vx",
    "GenPart_vy",
    "GenPart_vz",
    "GenProton_isPU",
    "GenProton_px",
    "GenProton_py",
    "GenProton_pz",
    "GenProton_vz",
]


def getDefaultColumnsToSave(isData):
    colToSave = defaultColToSave.copy()
    if not isData:
        colToSave.extend(["Pileup_nTrueInt"])
    return colToSave


def addAllVariables(
    dfw,
    syst_name,
    isData,
    trigger_class,
    lepton_legs,
    isSignal,
    applyTriggerFilter,
    global_params,
    channels,
    sample_name
):
    dfw.Apply(AnaBaseline.LeptonVeto)

    # dfw.Apply(AnaBaseline.RecoHttCandidateSelection, global_params)

    # dfw.Apply(AnaBaseline.JetSelection, global_params["era"])
    # dfw.Apply(AnaBaseline.JetSelection, global_params["era"])

    dfw.Apply(Corrections.getGlobal().jet.getEnergyResolution)

    dfw.Apply(Corrections.getGlobal().JetVetoMap.GetJetVetoMap)

    isV12 = (sample_name == "VBFHto2Mu_M-125_13p6TeV_powheg-herwig7" or sample_name== "VBFHto2Mu_M-125_13p6TeV_powheg-herwig7_ext1" or sample_name=="VBFHto2Mu_M-125_13p6TeV_powheg-herwig7_ext2")
    dfw.Apply(CommonBaseline.ApplyJetVetoMap, apply_filter=False,isV12=isV12)

    # dfw.Apply(AnaBaseline.GetMuMuCandidate)

    n_legs = 2

    for leg_idx in range(n_legs):

        def LegVar(
            var_name,
            var_expr,
            var_type=None,
            var_cond=None,
            check_leg_type=True,
            default=0,
        ):
            cond = var_cond
            # if check_leg_type:
            #     type_cond = f"HttCandidate.leg_type[{leg_idx}] != Leg::none"
            #     cond = f"{type_cond} && ({cond})" if cond else type_cond
            define_expr = (
                f"static_cast<{var_type}>({var_expr})" if var_type else var_expr
            )
            if cond:
                define_expr = f"{cond} ? ({define_expr}) : {default}"
            dfw.DefineAndAppend(f"mu{leg_idx+1}_{var_name}", define_expr)

        # LegVar("legType", f"HttCandidate.leg_type[{leg_idx}]", check_leg_type=False)
        LegVar("legType", f"Leg::mu", check_leg_type=False)
        for var in ["pt", "eta", "phi", "mass"]:
            LegVar(
                var,
                f"Muon_p4[mu{leg_idx+1}_idx].{var}()",
                var_type="float",
                default="-1000.f",
            )
        # fix: add muon index:

        LegVar(
            f"index",
            f"Muon_idx[mu{leg_idx+1}_idx]",
            var_type="int",
            default="-1",
        )

        LegVar("charge", f"Muon_charge[mu{leg_idx+1}_idx]", var_type="int")
        LegVar(
            f"pt_nano",
            f"static_cast<float>(Muon_p4_nano.at(mu{leg_idx+1}_idx).pt())",
        )

        for muon_obs in Muon_observables:
            if f"mu{leg_idx+1}_{muon_obs}" in dfw.df.GetColumnNames():
                continue
            if f"Muon_{muon_obs}" not in dfw.df.GetColumnNames():
                continue

            LegVar(
                muon_obs,
                f"Muon_{muon_obs}.at(mu{leg_idx+1}_idx)",
                # var_cond=f"HttCandidate.leg_type[{leg_idx}] == Leg::mu",
                default="-100000.f",
            )
        if not isData:
            # dfw.Define(
            #     f"mu{leg_idx+1}_genMatchIdx",
            #     f"HttCandidate.leg_type[{leg_idx}] != Leg::none ? HttCandidate.leg_genMatchIdx[{leg_idx}] : -1",
            # )
            LegVar(
                "gen_kind",
                f"genLeptons.at(mu{leg_idx+1}_genMatchIdx).kind()",
                var_type="int",
                var_cond=f"mu{leg_idx+1}_genMatchIdx>=0",
                default="static_cast<int>(GenLeptonMatch::NoMatch)",
            )
            LegVar(
                "gen_charge",
                f"genLeptons.at(mu{leg_idx+1}_genMatchIdx).charge()",
                var_type="int",
                var_cond=f"mu{leg_idx+1}_genMatchIdx>=0",
                default="-10",
            )
        else:
            # dfw.Define(
            #     f"mu{leg_idx+1}_genMatchIdx",
            #     f"-1",
            # )
            LegVar(
                "gen_kind",
                f"-1",
                var_type="int",
                default="-10",
            )
            LegVar(
                "gen_charge",
                f"-10",
                var_type="int",
                default="-10",
            )
        dfw.Define(
            f"mu{leg_idx+1}_p4",
            f"Muon_p4.at(mu{leg_idx+1}_idx)",
        )
        dfw.Define(
            f"mu{leg_idx+1}_p4_nano",
            f"Muon_p4_nano.at(mu{leg_idx+1}_idx)",
        )
    jet_obs_names = []
    for jvar in ["pt","eta","phi","mass"]:
        jet_obs_name = f"Jet_{jvar}"
        if f"{jet_obs_name}" in dfw.df.GetColumnNames():
            dfw.DefineAndAppend(f"{jet_obs_name}_nano",jet_obs_name)

    if not isData:
        JetObservables.extend(JetObservablesMC)
    for jetobs in JetObservables + ["idx"]:
        jet_obs_name = f"Jet_{jetobs}"
        if jet_obs_name in dfw.df.GetColumnNames():
            jet_obs_names.append(jet_obs_name)
    dfw.colToSave.extend(jet_obs_names)
    for recoObsNew in (
        PUObservables
        + FSRPhotonObservables
        + SoftActivityJetObservables
        + additional_VBFStudies_vars
    ):
        if recoObsNew in dfw.df.GetColumnNames():
            dfw.colToSave.extend([recoObsNew])

    # pf_str = global_params["met_type"]
    # dfw.DefineAndAppend(f"met_pt_nano", f"static_cast<float>({pf_str}_p4_nano.pt())")
    # dfw.DefineAndAppend(f"met_phi_nano", f"static_cast<float>({pf_str}_p4_nano.phi())")
    # dfw.DefineAndAppend("met_pt", f"static_cast<float>({pf_str}_p4.pt())")
    # dfw.DefineAndAppend("met_phi", f"static_cast<float>({pf_str}_p4.phi())")
    # dfw.DefineAndAppend("metnomu_pt_nano", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4_nano).pt())")
    # dfw.DefineAndAppend("metnomu_phi_nano", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4_nano).phi())")
    # dfw.DefineAndAppend("metnomu_pt", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4).pt())")
    # dfw.DefineAndAppend("metnomu_phi", f"static_cast<float>(GetMetNoMu(HttCandidate, {pf_str}_p4).phi())")
    # for var in ["covXX", "covXY", "covYY"]:
    #     dfw.DefineAndAppend(f"met_{var}", f"static_cast<float>({pf_str}_{var})")

    if trigger_class is not None:
        hltBranches = dfw.Apply(
            trigger_class.ApplyTriggers, lepton_legs, isData, applyTriggerFilter
        )
        dfw.colToSave.extend(hltBranches)

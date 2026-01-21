from FLAF.Common.Utilities import *


channels = ["muMu"]  # in order of importance during the channel selection
leg_names = ["Electron", "Muon", "Tau"]


def getChannelLegs(channel):
    ch_str = channel.lower()
    legs = []
    while len(ch_str) > 0:
        name_idx = None
        obj_name = None
        for idx, obj in enumerate(["e", "mu", "tau"]):
            if ch_str.startswith(obj):
                name_idx = idx
                obj_name = obj
                break
        if name_idx is None:
            raise RuntimeError(f"Invalid channel name {channel}")
        legs.append(leg_names[name_idx])
        ch_str = ch_str[len(obj_name) :]
    return legs


def RecoHttCandidateSelection(df, config):
    ####  COMPARISON WITH RUN2 ####
    # exactly two muons -- See AN/2019_185 line 118 and AN/2019_205 lines 246
    # these variables are needed to define the H(mumu) candidate structure
    df = df.Define("Muon_B2_muMu_1", "Muon_B0 && Muon_idx[Muon_B0].size()==2")
    df = df.Define("Muon_B2_muMu_2", "Muon_B0  && Muon_idx[Muon_B0].size()==2")
    cand_columns = []
    for ch in channels:
        leg1, leg2 = getChannelLegs(ch)
        cand_column = f"HttCandidates_{ch}"
        df = df.Define(
            cand_column,
            f"""
            GetHTTCandidates<2>(Channel::{ch}, 0., {leg1}_B2_{ch}_1, {leg1}_p4, {leg1}_iso, {leg1}_charge, {leg1}_genMatchIdx,{leg2}_B2_{ch}_2, {leg2}_p4, {leg2}_iso, {leg2}_charge, {leg2}_genMatchIdx)
        """,
        )
        cand_columns.append(cand_column)
    cand_filters = [f"{c}.size() > 0" for c in cand_columns]
    stringfilter = " || ".join(cand_filters)
    df = df.Filter(" || ".join(cand_filters), "Hmm candidate selection")
    cand_list_str = ", ".join(["&" + c for c in cand_columns])
    return df.Define(
        "HttCandidate", f"GetBestHTTCandidate<2>({{ {cand_list_str} }}, event)"
    )


def LeptonVeto(df, muon_pt_to_use="pt_nano"):
    df = df.Define("Muon_iso", "Muon_pfRelIso04_all")
    ####  COMPARISON WITH RUN2 ####
    # pT > 10 is a GENERAL preselection cut, then the muon matching to the offline one (which is the "leading" in Run2 analysis, in this case can be either the first or the second) has the offline pT threshold driven by the trigger. The eta, ID and iso cuts are the same w.r.t. Run 2 -- See AN/2019_185 lines 123 - 130 #  dxy < 0.5 cm, dz < 1.0 cm
    muon_sel_p4 = f"Muon_p4_{muon_pt_to_use}"
    print(f"Using muon p4: {muon_sel_p4} for lepton veto preselection")
    df = df.Define(
        "Muon_B0",
        f"""
        v_ops::pt({muon_sel_p4}) > 10 && abs(v_ops::eta({muon_sel_p4})) < 2.4 && (Muon_looseId && Muon_iso < 0.4)""",  #  loose id and very loose iso
        # v_ops::pt(Muon_p4) > 10 && abs(v_ops::eta(Muon_p4)) < 2.4 && (Muon_mediumId && Muon_iso < 0.25) && abs(Muon_dz) < 1. && abs(Muon_dxy) < 0.5""",
    )

    ####  COMPARISON WITH RUN2 ####
    # # exactly two muons -- See AN/2019_185 line 118 and AN/2019_205 lines 246
    df = df.Filter("Muon_idx[Muon_B0].size()==2", "No extra muons")

    ####  COMPARISON WITH RUN2 ####
    # Same electron selection w.r.t. Run 2 -- See AN/2019_185 lines 114 - 116
    df = df.Define(
        "Electron_B0_veto",
        f"""
        v_ops::pt(Electron_p4) > 20 && abs(v_ops::eta(Electron_p4)) < 2.5  && ( Electron_mvaIso_WP90 == true )""",
    )  # && abs(Electron_dz) < 0.2 && abs(Electron_dxy) < 0.024
    ####  COMPARISON WITH RUN2 ####
    # electron veto, same w.r.t. Run3 - See AN/2019_205 lines 246 - 248
    df = df.Filter("Electron_idx[Electron_B0_veto].size() == 0", "No extra electrons")
    df = df.Define("mu1_idx", "Muon_idx[Muon_B0][0]")
    df = df.Define("mu2_idx", "Muon_idx[Muon_B0][1]")
    return df


def JetSelection(df, era):
    # jet_puID_cut = ""
    # jet_puID_cut = "&& (Jet_puId>0 || v_ops::pt(Jet_p4)>50)" if era.startswith("Run2") else ""
    ####  COMPARISON WITH RUN2 ####
    # JetPUID  this is not applied because the PUID is no longer used in Run3
    df = df.Define(
        "Jet_B0", f"""v_ops::pt(Jet_p4) > 20 && abs(v_ops::eta(Jet_p4))< 4.7 """
    )
    ####  COMPARISON WITH RUN2 ####
    # pT > 20 is a GENERAL preselection cut, In Run2 it was set to 25 as default. See AN/2019_185 lines 89-91 . CAVEAT: In Run3 the PUPPI AK4 jets are used, not the CHS ones as in Run2
    df = df.Define(
        "Jet_B0p1", "Jet_B0 && ( Jet_jetId & 2 )" ""
    )  # loose jet ID as it was done in Run2, see AN/2019_185 lines 98-100

    df = df.Define("JetSel", "Jet_idx[Jet_B0p1].size()>0")
    # IN THIS CASE, NO FILTER IS APPLIED. This is the Jet Selection definition, but it will be used for later cuts.
    return df


def GetMuMuCandidate(df):
    df = df.Define("Hmumu_idx", "Muon_idx[Muon_B0]")
    return df


"""

def getChannelLegs(channel):
    ch_str = channel.lower()
    legs = []
    while len(ch_str) > 0:
        name_idx = None
        obj_name = None
        for idx, obj in enumerate(['e', 'mu', 'tau']):
            if ch_str.startswith(obj):
                name_idx = idx
                obj_name = obj
                break
        if name_idx is None:
            raise RuntimeError(f"Invalid channel name {channel}")
        legs.append(leg_names[name_idx])
        ch_str = ch_str[len(obj_name):]
    return legs

def PassGenAcceptance(df):
    df = df.Filter("genHttCandidate.get() != nullptr", "genHttCandidate present")
    return df.Filter("PassGenAcceptance(*genHttCandidate)", "genHttCandidate Acceptance")

def GenJetSelection(df):
    df = df.Define("GenJet_B1","GenJet_pt > 20 && abs(GenJet_eta) < 2.5 && GenJet_Hbb")
    df = df.Define("GenJetAK8_B1","GenJetAK8_pt > 170 && abs(GenJetAK8_eta) < 2.5 && GenJetAK8_Hbb")
    return df.Filter("GenJet_idx[GenJet_B1].size()==2 || (GenJetAK8_idx[GenJetAK8_B1].size()==1 && genHbb_isBoosted)", "(One)Two b-parton (Fat)jets at least")

def GenJetHttOverlapRemoval(df):
    for var in ["GenJet", "GenJetAK8"]:
        df = df.Define(f"{var}_B2", f"RemoveOverlaps({var}_p4, {var}_B1,{{{{genHttCandidate->leg_p4[0], genHttCandidate->leg_p4[1]}},}}, 2, 0.5)" )
    return df.Filter("GenJet_idx[GenJet_B2].size()==2 || (GenJetAK8_idx[GenJetAK8_B2].size()==1 && genHbb_isBoosted)", "No overlap between genJets and genHttCandidates")

def RequestOnlyResolvedGenJets(df):
    return df.Filter("GenJet_idx[GenJet_B2].size()==2", "Resolved topology")
"""

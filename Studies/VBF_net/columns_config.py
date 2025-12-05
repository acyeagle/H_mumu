
columns_config = {
    "metadata" : [
        "FullEventId",
        "isData",
        #"weight_MC_Lumi_pu",
        "final_weight",
        "era",
        "process",
        "dataset",
        "baseline_muonJet",
        "Signal_Fit",
        "Signal_ext",
        "H_sideband",
        "Z_sideband",
        "VBF",
        "VBF_JetVeto"
    ],
    "flat_vars" : [
        "pt_mumu",
        "m_mumu",
        "mu1_eta",
        "mu2_eta",
        "nSoftActivityJet",
        "SoftActivityJetHT",
        "SoftActivityJetHT2",
        "SoftActivityJetHT5",
        "SoftActivityJetHT10",
        "SoftActivityJetNjets2",
        "SoftActivityJetNjets5",
        "SoftActivityJetNjets10",
        "nJet"
    ],
    "jet_vars" : [
        'FilteredJet_pt',
        'FilteredJet_eta',
        'FilteredJet_phi',
        'FilteredJet_btagPNetB',
        'FilteredJet_btagPNetCvB',
        'FilteredJet_btagPNetCvL',
        'FilteredJet_btagPNetCvNotB',
        'FilteredJet_btagPNetQvG',
        'FilteredJet_btagPNetTauVJet'
    ]
}
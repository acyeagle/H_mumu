import math
import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])



ROOT.gInterpreter.Declare(
    """
    float get_scale_factor_error(const float& effData, const float& effMC, const float& errData, const float& errMC, std::string err_name) {

    float SF_error = 0.;

    if (errData==0. && errMC==0.) {
        //std::cout<<"WARNING : uncertainty on data and MC = 0, for " << err_name << ", can not calculate uncertainty on scale factor. Uncertainty set to 0." << std::endl;
        return 0.;
    }

    if (effData==0. || effMC==0.) {
        //std::cout<<"WARNING : uncertainty on OR and MC = 0, for " << err_name << ",can not calculate uncertainty on scale factor. Uncertainty set to 0." << std::endl;
        return 0.;
    }
    else {
        SF_error = pow((errData/effData),2) + pow((errMC/effMC),2);
        SF_error = pow(SF_error, 0.5)*(effData/effMC);
    }
    return SF_error;
    }

    float SelectCorrectDM(const float& tauDM, const float& effDM0, const float& effDM1, const float& eff3Prong){
        if(tauDM == 0){
            return effDM0;
        }
        if(tauDM == 1){
            return effDM1;
        }
        if(tauDM == 10 || tauDM == 11){
            return eff3Prong;
        }

        return 0.;
    }
    float getCorrectSingleLepWeight(const float& lep1_pt, const float& lep1_eta, const bool& lep1_matching, const float& lep1_weight,const float& lep2_pt, const float& lep2_eta, const bool& lep2_matching, const float& lep2_weight){
        if(lep1_pt > lep2_pt){
            return lep1_matching? lep1_weight : 1.f;
        }
        else if(lep1_pt == lep2_pt){
            if(abs(lep1_eta) < abs(lep2_eta)){
                return lep1_matching? lep1_weight : 1.f;
            }
            else if(abs(lep1_eta) > abs(lep2_eta)){
                return lep2_matching? lep2_weight : 1.f;
            }
        }
        else{
            return lep2_matching? lep2_weight : 1.f;
        }
    throw std::invalid_argument("ERROR: no suitable single lepton candidate");

    }
    """
)


def get_scale_factor_error(eff_data, eff_mc, err_data, err_mc):
    SF_error = 0.0

    if err_data == 0.0 and err_mc == 0.0:
        print(
            "WARNING: uncertainty on data and MC = 0, cannot calculate uncertainty on scale factor. Uncertainty set to 0."
        )

    if eff_data == 0.0 or eff_mc == 0.0:
        print(
            "WARNING: efficiency in data or MC = 0, cannot calculate uncertainty on scale factor. Uncertainty set to 0."
        )
        return 0.0
    else:
        SF_error = math.pow((err_data / eff_data), 2) + math.pow((err_mc / eff_mc), 2)
        SF_error = math.sqrt(SF_error) * (eff_data / eff_mc)

    return SF_error


def defineTriggerWeights(dfBuilder):  # needs application region def
    dfBuilder.df = dfBuilder.df.Define(
        f"weight_trigSF_singleMu",
        "if (HLT_singleMu && muMu) {return getCorrectSingleLepWeight(mu1_pt, mu1_eta, mu1_HasMatching_singleMu, weight_mu1_TrgSF_singleMu_Central,mu2_pt, mu2_eta, mu2_HasMatching_singleMu, weight_mu2_TrgSF_singleMu_Central) ;} return 1.f;",
    )


def defineTriggerWeightsErrors(dfBuilder):
    for scale in ["Up", "Down"]:
        trg_name = "singleMu_IsoMu24"
        dfBuilder.df = dfBuilder.df.Define(
            f"weight_trigSF_{trg_name}{scale}_rel",
            f"""if (HLT_singleMu && muMu) {{return getCorrectSingleLepWeight(mu1_pt, mu1_eta, mu1_HasMatching_singleMu, weight_mu1_TrgSF_{trg_name}{scale}_rel,mu2_pt, mu2_eta, mu2_HasMatching_singleMu, weight_mu2_TrgSF_{trg_name}{scale}_rel) ;}} return 1.f;""",
        )


def AddTriggerWeightsAndErrors(dfBuilder, WantErrors):
    defineTriggerWeights(dfBuilder)
    if WantErrors:
        defineTriggerWeightsErrors(dfBuilder)


# def defineTauTauTriggerWeightsErrors(dfBuilder):
#     for scale in ['Up', 'Down']:
#         ### diTau - for tauTau ###
#         dfBuilder.df = dfBuilder.df.Define(
#             f"trigSF_tau{scale}_rel",
#             f"if (HLT_ditau && Legacy_region && tauTau) {{return (SelectCorrectDM(tau1_decayMode, weight_tau1_TrgSF_ditau_DM0{scale}_rel, weight_tau1_TrgSF_ditau_DM1{scale}_rel, weight_tau1_TrgSF_ditau_3Prong{scale}_rel)*SelectCorrectDM(tau2_decayMode, weight_tau2_TrgSF_ditau_DM0{scale}_rel, weight_tau2_TrgSF_ditau_DM1{scale}_rel, weight_tau2_TrgSF_ditau_3Prong{scale}_rel)); }}return 1.f;"
#         )
#         ### singleTau ###
#         dfBuilder.df = dfBuilder.df.Define(f"weight_trigSF_singleTau{scale}_rel", f"""if (HLT_singleTau && (tauTau ) && SingleTau_region && !(Legacy_region)) {{return getCorrectSingleLepWeight(tau1_pt, tau1_eta, tau1_HasMatching_singleTau, weight_tau1_TrgSF_singleTau{scale}_rel,tau2_pt, tau2_eta, tau2_HasMatching_singleTau, weight_tau2_TrgSF_singleTau{scale}_rel); }} else if (HLT_singleTau && (eTau || muTau ) && SingleTau_region && !(Legacy_region)) {{return weight_tau2_TrgSF_singleTau{scale}_rel;}} ;return 1.f;""")
#         ### MET ###
#         dfBuilder.df = dfBuilder.df.Define(f"weight_trigSF_MET{scale}_rel", f"if(HLT_MET && !(SingleTau_region) && !(Legacy_region)) {{return weight_TrgSF_MET{scale}_rel;}} return 1.f;")
#         #### final tau trig sf ##### --> not needed as there is only tauTau channel
#         # dfBuilder.df = dfBuilder.df.Define(f"trigSF_tau{scale}_rel", f"""if (Legacy_region && eTau){{return eTau_trigSF_tau{scale}_rel;}} else if (Legacy_region && muTau){{return muTau_trigSF_tau{scale}_rel;}} else if(Legacy_region && tauTau){{return tauTau_trigSF_tau{scale}_rel;}} return 1.f;""")


# def defineTauTauTriggerWeights(dfBuilder): # needs application region def
#     # *********************** tauTau ***********************
#     if 'tauTau' in dfBuilder.config['channels_to_consider']:
#         dfBuilder.df = dfBuilder.df.Define(
#                     f"weight_trigSF_diTau",
#                     f"if (HLT_ditau && Legacy_region && tauTau) {{return (SelectCorrectDM(tau1_decayMode, weight_tau1_TrgSF_ditau_DM0Central, weight_tau1_TrgSF_ditau_DM1Central, weight_tau1_TrgSF_ditau_3ProngCentral)*SelectCorrectDM(tau2_decayMode, weight_tau2_TrgSF_ditau_DM0Central, weight_tau2_TrgSF_ditau_DM1Central, weight_tau2_TrgSF_ditau_3ProngCentral)); }}return 1.f;"
#                 )
#     # *********************** singleTau ***********************

#     # dfBuilder.df = dfBuilder.df.Define(f"weight_trigSF_singleTau", f"if  (HLT_singleTau && (tauTau || eTau || muTau ) && SingleTau_region && !(Legacy_region)) {{return getCorrectSingleLepWeight(tau1_pt, tau1_eta, tau1_HasMatching_singleTau, weight_tau1_TrgSF_singleTauCentral,tau2_pt, tau2_eta, tau2_HasMatching_singleTau, weight_tau2_TrgSF_singleTauCentral);}} return 1.f;")
#     # if 'tauTau' in dfBuilder.config['channels_to_consider'] or 'muTau' in dfBuilder.config['channels_to_consider'] or 'eTau' in dfBuilder.config['channels_to_consider']  :
#         dfBuilder.df = dfBuilder.df.Define(f"weight_trigSF_singleTau", f"""if (HLT_singleTau && (tauTau ) && SingleTau_region && !(Legacy_region)) {{return getCorrectSingleLepWeight(tau1_pt, tau1_eta, tau1_HasMatching_singleTau, weight_tau1_TrgSF_singleTauCentral,tau2_pt, tau2_eta, tau2_HasMatching_singleTau, weight_tau2_TrgSF_singleTauCentral); }} else if (HLT_singleTau && (eTau || muTau ) && SingleTau_region && !(Legacy_region)) {{return weight_tau2_TrgSF_singleTauCentral;}} ;return 1.f;""")
#     # *********************** MET ***********************
#     # if 'tauTau' in dfBuilder.config['channels_to_consider'] or 'muTau' in dfBuilder.config['channels_to_consider'] or 'eTau' in dfBuilder.config['channels_to_consider']  :
#         dfBuilder.df = dfBuilder.df.Define(f"weight_trigSF_MET", "if (HLT_MET && (tauTau || eTau || muTau ) && !(SingleTau_region) && !(Legacy_region)) { return (weight_TrgSF_METCentral) ;} return 1.f;")

# def AddTauTauTriggerWeightsAndErrors(dfBuilder,WantErrors):
#     defineTauTauTriggerWeights(dfBuilder)
#     if WantErrors:
#         defineTauTauTriggerWeightsErrors(dfBuilder)

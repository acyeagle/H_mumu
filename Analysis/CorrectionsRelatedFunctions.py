import ROOT
import correctionlib
import os

correctionlib.register_pyroot_binding()

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.Utilities import *
ROOT.gInterpreter.Declare(
    """
    float GetGenPtLL(const ROOT::VecOps::RVec<float> &GenPart_pt,
                    const ROOT::VecOps::RVec<float> &GenPart_phi,
                    const ROOT::VecOps::RVec<float> &GenPart_eta,
                    const ROOT::VecOps::RVec<float> &GenPart_mass,
                    const ROOT::VecOps::RVec<int>   &GenPart_pdgId,
                    const ROOT::VecOps::RVec<unsigned short> &GenPart_statusFlags,
                    const ROOT::VecOps::RVec<int>   &GenPart_status)
    {
        using lorentzvector = ROOT::Math::PtEtaPhiMVector;

        int part1_idx = -1;
        int part2_idx = -1;

        // Bit 8 of statusFlags = "isHardProcess"
        const unsigned short hardProcessMask = (1 << 8);

        for (size_t i = 0; i < GenPart_pdgId.size(); i++) {

            bool isHardProcess = (GenPart_statusFlags[i] & hardProcessMask);
            int pdg = std::abs(GenPart_pdgId[i]);

            if ( ((pdg == 11 || pdg == 13 ) && GenPart_status[i]==1 || (pdg == 15 && GenPart_status[i]==2)) && isHardProcess ) {
                if (part1_idx < 0)
                    part1_idx = i;
                else if (part1_idx >= 0 && part2_idx < 0)
                    part2_idx = i;
                else
                    std::cout << "WARNING: più di due leptoni hard process trovati"<<std::endl;
            }
        }

        // Controlli
        if (part1_idx < 0 || part2_idx < 0) {
            // Nessuna coppia trovata
            return -1.0;
        }

        lorentzvector p1(
            GenPart_pt[part1_idx],
            GenPart_eta[part1_idx],
            GenPart_phi[part1_idx],
            GenPart_mass[part1_idx]
        );

        lorentzvector p2(
            GenPart_pt[part2_idx],
            GenPart_eta[part2_idx],
            GenPart_phi[part2_idx],
            GenPart_mass[part2_idx]
        );

        return (p1 + p2).Pt();
    }

    """
)
from Analysis.GetTriggerWeights import *


def RedefineIsoTrgAndIDWeights(df, period):
    correctionlib.register_pyroot_binding()

    year = period.split("_")[1]
    analysis_path = os.environ["ANALYSIS_PATH"]
    year_dict = {
        "Run3_2022":"2022_Summer22",
        "Run3_2022EE":"2022_Summer22EE",
        "Run3_2023":"2023_Summer23",
        "Run3_2023BPix":"2023_Summer23BPix",
    }
    ROOT.gROOT.ProcessLine(
        f'auto cset = correction::CorrectionSet::from_file("/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/POG/MUO/{year_dict[period]}/muon_Z.json.gz");'
    )
    for muon_idx in [1,2]:
        # NUM_TightPFIso_DEN_TightID --> Iso
        # NUM_TightID_DEN_TrackerMuons --> ID
        # NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight --> Trg

        ### new tight ID - tight iso weights ####
        df = df.Define(f"""weight_mu{muon_idx}_tightID""",f"""mu{muon_idx}_pt_nano > 15 ?cset->at("NUM_TightID_DEN_TrackerMuons")->evaluate({{mu{muon_idx}_eta, mu{muon_idx}_pt_nano, "nominal"}}) : 1.f""")
        df = df.Define(f"""weight_mu{muon_idx}_tightID_tightIso""",f"""mu{muon_idx}_pt_nano > 15 ?cset->at("NUM_TightPFIso_DEN_TightID")->evaluate({{mu{muon_idx}_eta, mu{muon_idx}_pt_nano, "nominal"}}) : 1.f""")
        df = df.Define(f"""weight_mu{muon_idx}_TRG_tightID_tightIso""",f"""mu{muon_idx}_pt_nano > 26 ?cset->at("NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight")->evaluate({{mu{muon_idx}_eta, mu{muon_idx}_pt_nano, "nominal"}}) : 1.f""")

        ### new medium ID - loose/medium iso weights ####
        # NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium
        # NUM_MediumID_DEN_TrackerMuons
        # NUM_LoosePFIso_DEN_MediumID
        df = df.Define(f"""weight_mu{muon_idx}_mediumID""",f"""mu{muon_idx}_pt_nano > 15 ?cset->at("NUM_MediumID_DEN_TrackerMuons")->evaluate({{mu{muon_idx}_eta, mu{muon_idx}_pt_nano, "nominal"}}) : 1.f""")
        df = df.Define(f"""weight_mu{muon_idx}_mediumID_looseIso""",f"""mu{muon_idx}_pt_nano > 15 ?cset->at("NUM_LoosePFIso_DEN_MediumID")->evaluate({{mu{muon_idx}_eta, mu{muon_idx}_pt_nano, "nominal"}}) : 1.f""")
        df = df.Define(f"""weight_mu{muon_idx}_TRG_mediumID_mediumIso""",f"""mu{muon_idx}_pt_nano > 26 ?cset->at("NUM_IsoMu24_DEN_CutBasedIdMedium_and_PFIsoMedium")->evaluate({{mu{muon_idx}_eta, mu{muon_idx}_pt_nano, "nominal"}}) : 1.f""")

    df = df.Define(f"weight_trigSF_singleMu_tightID_tightIso",
        "if (HLT_singleMu && muMu) {return getCorrectSingleLepWeight(mu1_pt_nano, mu1_eta, mu1_HasMatching_singleMu, weight_mu1_TRG_tightID_tightIso,mu2_pt_nano, mu2_eta, mu2_HasMatching_singleMu, weight_mu1_TRG_tightID_tightIso) ;} return 1.f;",
        )
    df = df.Define(f"weight_trigSF_singleMu_mediumID_mediumIso",
        "if (HLT_singleMu && muMu) {return getCorrectSingleLepWeight(mu1_pt_nano, mu1_eta, mu1_HasMatching_singleMu, weight_mu1_TRG_mediumID_mediumIso,mu2_pt_nano, mu2_eta, mu2_HasMatching_singleMu, weight_mu1_TRG_mediumID_mediumIso) ;} return 1.f;",
        )
    return df


def AddNewDYWeights(df, period, isDY):
    correctionlib.register_pyroot_binding()
    for idx in [0,1]:
        df = df.Define(
            f"mu{idx+1}_p4_nano",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt_nano,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )

    year = period.split("_")[1]
    analysis_path = os.environ["ANALYSIS_PATH"]
    year_dict = {
        "Run3_2022":"2022preEE",
        "Run3_2022EE":"2022postEE",
        "Run3_2023":"2023preBPix",
        "Run3_2023BPix":"2023postBPix",
    }
    if isDY:
        ROOT.gROOT.ProcessLine(
            f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Corrections/data/hleprare/DYweightCorrlib/DY_pTll_weights_{year_dict[period]}_v5.json.gz");'
        )
        df = df.Define("pt_ll_nano","""(mu1_p4_nano + mu2_p4_nano).pt()""")
        df = df.Define("genpt_ll","""GetGenPtLL( GenPart_pt, GenPart_phi, GenPart_eta, GenPart_mass, GenPart_pdgId, GenPart_statusFlags, GenPart_status)""")
        sample_order = '"NLO"'

        df = df.Define("newDYWeight_ptLL_nano",f"""return pt_ll_nano >= 0 ? cset->at("DY_pTll_reweighting")->evaluate({{ {sample_order}, pt_ll_nano, "nom"}}) : 1.f""")
        # df = df.Define("newDYWeight_ptLL_ScaRe",f"""return pt_ll_nano >= 0 ? cset->at("DY_pTll_reweighting")->evaluate({{ {sample_order}, pt_ll_nano, "nom"}}) : 1.f""")
        df = df.Define("newDYWeight_genpt_ll",f"""return genpt_ll >= 0 ? cset->at("DY_pTll_reweighting")->evaluate({{ {sample_order}, genpt_ll, "nom"}}) : 1.f""")
    else:
        df = df.Define("newDYWeight_ptLL_nano","""1.f""")
        df = df.Define("newDYWeight_genpt_ll","""1.f""")
        # df = df.Define("newDYWeight","""1.f""")
        # df = df.Define("newDYWeight","""1.f""")

    return df


def AddRoccoR(df, period, isData):
    year = period.split("_")[1]
    analysis_path = os.environ["ANALYSIS_PATH"]
    ROOT.gROOT.ProcessLine(
        f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Corrections/data/MUO/RoccoR/RoccoR{year}.json.gz");'
    )
    ROOT.gROOT.ProcessLine(f'#include "{analysis_path}/include/RoccoR.cc"')
    for mu_idx in [1, 2]:
        if isData:
            # Data apply scale correction
            df = df.Define(
                f"mu{mu_idx}_RoccoR_scale",
                f"""cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}})"""
                # f"""if(mu{mu_idx}_pt_nano >= 20) return cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}); return -1."""
            ).Define(f"mu{mu_idx}_RoccoR_pt",f"mu{mu_idx}_pt_nano * mu{mu_idx}_RoccoR_scale")
            df = df.Define(
                f"mu{mu_idx}_BS_RoccoR_scale",
                # f"""if(mu{mu_idx}_bsConstrainedPt >= 20) return cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}); return -1."""

                f"""cset->at("kScaleDT")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}})"""
            ).Define(f"mu{mu_idx}_BS_RoccoR_pt",f"mu{mu_idx}_bsConstrainedPt * mu{mu_idx}_BS_RoccoR_scale")
        else:
            df = df.Define(f"genmu{mu_idx}_pT", f"mu{mu_idx}_genPartIdx >= 0 ? GenPart_pt.at(mu{mu_idx}_genPartIdx) : -1.")
            # df.Display({f"mu{mu_idx}_charge"}).Print()
            df = df.Define(
                f"mu{mu_idx}_RoccoR_scale",
                f""" mu{mu_idx}_genPartIdx >= 0? cset->compound().at("kSpreadMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi,  genmu{mu_idx}_pT, 0, 0}}) : cset->at("kScaleMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}) """
            ).Define(f"mu{mu_idx}_RoccoR_pt",f"mu{mu_idx}_pt_nano * mu{mu_idx}_RoccoR_scale")

            df = df.Define(
                f"mu{mu_idx}_BS_RoccoR_scale",
                f""" mu{mu_idx}_genPartIdx >= 0? cset->compound().at("kSpreadMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi,  genmu{mu_idx}_pT, 0, 0}}) : cset->at("kScaleMC")->evaluate({{mu{mu_idx}_charge, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, 0, 0}}) """
            ).Define(f"mu{mu_idx}_BS_RoccoR_pt",f"mu{mu_idx}_bsConstrainedPt * mu{mu_idx}_BS_RoccoR_scale")
    return df


def AddScaReOnBS(df, period, isData):
    import correctionlib

    period_files = {
        "Run3_2022": "2022_Summer22",
        "Run3_2022EE": "2022_Summer22EE",
        "Run3_2023": "2023_Summer23",
        "Run3_2023BPix": "2023_Summer23BPix",
    }
    correctionlib.register_pyroot_binding()
    file_name = period_files.get(period, "")
    analysis_path = os.environ["ANALYSIS_PATH"]
    ROOT.gROOT.ProcessLine(
        # f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Analysis/schemaV2_VXBS.json");' # only for 2024
        f'auto cset = correction::CorrectionSet::from_file("{analysis_path}/Corrections/data/MUO/MuonScaRe/{file_name}.json");'
    )
    ROOT.gROOT.ProcessLine(f'#include "{analysis_path}/include/MuonScaRe.cc"')
    for mu_idx in [1, 2]:
        if isData:
            # Data apply scale correction
            df = df.Define(
                f"mu{mu_idx}_BS_pt_1_corr",
                f"pt_scale(1, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
            )
            df = df.Define(
                f"mu{mu_idx}_reapplied_pt_1_corr",
                f"pt_scale(1, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
            )
        else:
            df = df.Define(
                f"mu{mu_idx}_BS_pt_1_scale_corr",
                f"pt_scale(0, mu{mu_idx}_bsConstrainedPt, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
            )

            df = df.Define(
                f"mu{mu_idx}_BS_pt_1_corr",
                f"pt_resol(mu{mu_idx}_BS_pt_1_scale_corr, mu{mu_idx}_eta, float(mu{mu_idx}_nTrackerLayers))",
            )

            df = df.Define(
                f"mu{mu_idx}_reapplied_pt_1_scale_corr",
                f"pt_scale(0, mu{mu_idx}_pt_nano, mu{mu_idx}_eta, mu{mu_idx}_phi, mu{mu_idx}_charge)",
            )
            df = df.Define(
                f"mu{mu_idx}_reapplied_pt_1_corr",
                f"pt_resol(mu{mu_idx}_reapplied_pt_1_scale_corr, mu{mu_idx}_eta, float(mu{mu_idx}_nTrackerLayers))",
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
    # df.Filter("event==74738519").Display({"event",f"mu1_pt_nano","mu2_pt_nano"}).Print()
    # df.Filter("event==74738519").Display({"event",f"mu1_reapplied_pt_1_corr","mu2_reapplied_pt_1_corr"}).Print()
    # df.Filter("event==74738519").Display({f"mu1_reapplied_pt_1_corr","mu2_reapplied_pt_1_corr"}).Print()
    return df




def RescaleXS(df, config):
    import yaml
    xsFile = config["crossSectionsFile"]
    xsFilePath = os.path.join(os.environ["ANALYSIS_PATH"], xsFile)
    with open(xsFilePath, "r") as xs_file:
        xs_dict = yaml.safe_load(xs_file)
    xs_condition = f"DY" in config["process_name"] #== "DY"
    xs_to_scale = (
        xs_dict["DY_NNLO_QCD+NLO_EW"]["crossSec"] if xs_condition else "1.f"
    )
    weight_XS_string = f"xs_to_scale/current_xs" if xs_condition else "1."
    total_denunmerator_nJets = 5378.0 / 3 + 1017.0 / 3 + 385.5 / 3
    df = df.Define(f"current_xs", f"{total_denunmerator_nJets}")
    df = df.Define(f"xs_to_scale", f"{xs_to_scale}")
    df = df.Define(f"weight_XS", weight_XS_string)
    return df
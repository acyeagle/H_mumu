import ROOT

if __name__ == "__main__":
    sys.path.append(os.environ["ANALYSIS_PATH"])


from FLAF.Common.Utilities import *
def GetMuMuMassResolution(df):
    delta_mu_expr = "sqrt( 0.5 * (pow( ({0}/{1}), 2) + pow( ({2}/{3}), 2) ) ) "
    df = df.Define(
        "m_mumu_resolution_nano",
        delta_mu_expr.format(
            "mu1_ptErr",
            "mu1_pt_nano",
            "mu2_ptErr",
            "mu2_pt_nano",
        ),
    )
    df = df.Define(
        "m_mumu_resolution",
        delta_mu_expr.format(
            "(mu1_pt-mu1_pt_nano)/mu1_pt",
            "mu1_pt",
            "(mu2_pt-mu2_pt_nano)/mu2_pt",
            "mu2_pt",
        ),
    )
    df = df.Define(
        "m_mumu_resolution_BS",
        delta_mu_expr.format(
            "mu1_bsConstrainedPtErr",
            "mu1_bsConstrainedPt",
            "mu2_bsConstrainedPtErr",
            "mu2_bsConstrainedPt",
        ),
    )
    df = df.Define(
        "m_mumu_resolution_BS_ScaRe",
        delta_mu_expr.format(
            "(mu1_BS_pt_1_corr-mu1_bsConstrainedPt)",
            "mu1_BS_pt_1_corr",
            "(mu2_BS_pt_1_corr-mu2_bsConstrainedPt)",
            "mu2_BS_pt_1_corr",
        ),
    )

    return df


def GetAllMuMuPtRelatedObservables(df):
    for idx in [0, 1]:
        df = defineP4(df, f"mu{idx+1}")
        df = df.Define(
            f"mu{idx+1}_p4_reapplied",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_reapplied_pt_1_corr,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
        df = df.Define(
            f"mu{idx+1}_p4_BS",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_bsConstrainedPt,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
        if f"mu{idx+1}_p4_nano" not in df.GetColumnNames():
            df = df.Define(
                f"mu{idx+1}_p4_nano",
                f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt_nano,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
            )
        df = df.Define(
            f"mu{idx+1}_p4_BS_ScaRe",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_BS_pt_1_corr,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )

        df = df.Define(
            f"mu{idx+1}_p4_RoccoR",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_RoccoR_pt,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
        df = df.Define(
            f"mu{idx+1}_p4_BS_RoccoR",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_BS_RoccoR_pt,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
    df = df.Define(f"pt_mumu_reapplied", "(mu1_p4_reapplied+mu2_p4_reapplied).Pt()")
    df = df.Define(f"pt_mumu_BS", "(mu1_p4_BS+mu2_p4_BS).Pt()")
    df = df.Define(f"pt_mumu_nano", "(mu1_p4_nano+mu2_p4_nano).Pt()")
    df = df.Define(f"pt_mumu_BS_ScaRe", "(mu1_p4_BS_ScaRe+mu2_p4_BS_ScaRe).Pt()")
    df = df.Define(f"pt_mumu_RoccoR", "(mu1_p4_RoccoR+mu2_p4_RoccoR).Pt()")
    df = df.Define(f"pt_mumu_BS_RoccoR", "(mu1_p4_BS_RoccoR+mu2_p4_BS_RoccoR).Pt()")


    df = df.Define(f"m_mumu_reapplied", "(mu1_p4_reapplied+mu2_p4_reapplied).M()")
    df = df.Define(f"m_mumu_BS", "(mu1_p4_BS+mu2_p4_BS).M()")
    df = df.Define(f"m_mumu_nano", "(mu1_p4_nano+mu2_p4_nano).M()")
    df = df.Define(f"m_mumu_BS_ScaRe", "(mu1_p4_BS_ScaRe+mu2_p4_BS_ScaRe).M()")
    df = df.Define(f"m_mumu_RoccoR", "(mu1_p4_RoccoR+mu2_p4_RoccoR).M()")
    df = df.Define(f"m_mumu_BS_RoccoR", "(mu1_p4_BS_RoccoR+mu2_p4_BS_RoccoR).M()")
    return df

def RedefineMuonsPt(df, pt_to_use):
    pt_names_dict = {
        "nano":"pt_nano",
        "scare":"pt",
        "scare_reapplied":"reapplied_pt_1_corr", # done 1
        "BS":"bsConstrainedPt",
        "BS_scare":"BS_pt_1_corr",
        "RoccoR":"RoccoR_pt",
        "BS_RoccoR":"BS_RoccoR_pt"
    }
    pt_suffix = pt_names_dict[pt_to_use]
    print(f"using the pT: {pt_suffix}")
    for idx in [0,1]:
        df = df.Redefine(f"mu{idx+1}_pt",f"mu{idx+1}_{pt_suffix}")
        df = df.Redefine(
            f"mu{idx+1}_p4",
            f"ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>(mu{idx+1}_pt,mu{idx+1}_eta,mu{idx+1}_phi,mu{idx+1}_mass)",
        )
    return df

def RedefineDiMuonObservables(df):
    df = df.Define(f"pt_mumu", "(mu1_p4+mu2_p4).Pt()")
    df = df.Define(f"y_mumu", "(mu1_p4+mu2_p4).Rapidity()")
    df = df.Define(f"eta_mumu", "(mu1_p4+mu2_p4).Eta()")
    df = df.Define(f"phi_mumu", "(mu1_p4+mu2_p4).Phi()")
    df = df.Define("m_mumu", "static_cast<float>((mu1_p4+mu2_p4).M())")

    for idx in [0, 1]:
        df = df.Define(f"mu{idx+1}_pt_rel", f"mu{idx+1}_pt/m_mumu")

    df = df.Define("dR_mumu", "ROOT::Math::VectorUtil::DeltaR(mu1_p4, mu2_p4)")

    df = df.Define("Ebeam", "13600.0/2")
    df = df.Define("cosTheta_Phi_CS", "ComputeCosThetaPhiCS(mu1_p4, mu2_p4,  Ebeam)")
    df = df.Define("cosTheta_CS", "static_cast<float>(std::get<0>(cosTheta_Phi_CS))")
    df = df.Define("phi_CS", "static_cast<float>(std::get<1>(cosTheta_Phi_CS))")
    return df
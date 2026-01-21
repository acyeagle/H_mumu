#!/usr/bin/env python3
import uproot
import numpy as np
import json
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
args = parser.parse_args()


def load(fname):
    f = uproot.open(fname)["Mini"]
    return {k: f[k].array(library="np") for k in f.keys()}


sig = load(
    f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/mini_signal_{args.year}.root"
)
bkg = load(
    f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/mini_bkgTTbar_{args.year}.root"
)

isoWPs = [0.25, 0.20, 0.15]
idWPs = ["loose", "medium", "tight"]
regions = ["baseline_noID_Iso", "VBF_JetVeto_noID_Iso", "ggH_noID_Iso"]


def compute_eff(data, wp1, wp2, idWP_1, idWP_2, region):
    mask = (
        data["Signal_Fit"]
        & data[region]
        & data[f"mu1_{idWP_1}Id"]
        & data[f"mu2_{idWP_2}Id"]
        & (data["mu1_pfRelIso04_all"] < wp1)
        & (data["mu2_pfRelIso04_all"] < wp2)
    )
    total_mask = data["Signal_Fit"] & data[region]

    w_sel = data["weight_MC_Lumi_pu"][mask]
    w_total = data["weight_MC_Lumi_pu"][total_mask]

    eff = w_sel.sum() / max(w_total.sum(), 1)

    eff_err = np.sqrt(np.sum(w_sel**2)) / max(w_total.sum(), 1)

    return {
        "count": int(w_sel.sum()),
        "total": int(w_total.sum()),
        "eff": float(eff),
        "err": float(eff_err),
    }


def compute_ssqrtb(data_sig, data_bckg, wp1, wp2, idWP_1, idWP_2, region):
    mask_sig = (
        data_sig["Signal_Fit"]
        & data_sig[region]
        & data_sig[f"mu1_{idWP_1}Id"]
        & data_sig[f"mu2_{idWP_2}Id"]
        & (data_sig["mu1_pfRelIso04_all"] < wp1)
        & (data_sig["mu2_pfRelIso04_all"] < wp2)
    )
    mask_bkg = (
        data_bckg["Signal_Fit"]
        & data_bckg[region]
        & data_bckg[f"mu1_{idWP_1}Id"]
        & data_bckg[f"mu2_{idWP_2}Id"]
        & (data_bckg["mu1_pfRelIso04_all"] < wp1)
        & (data_bckg["mu2_pfRelIso04_all"] < wp2)
    )

    w_sig = data_sig["weight_MC_Lumi_pu"][mask_sig]
    w_bkg = data_bckg["weight_MC_Lumi_pu"][mask_bkg]

    S = w_sig.sum()
    B = w_bkg.sum()
    sigma_S = np.sqrt(np.sum(w_sig**2))
    sigma_B = np.sqrt(np.sum(w_bkg**2))

    s_sqrtB = S / np.sqrt(max(B, 1))
    s_sqrtB_err = np.sqrt(
        (sigma_S / np.sqrt(max(B, 1))) ** 2 + (S * sigma_B / (2 * B ** (3 / 2))) ** 2
    )

    return {"s_sqrtB": float(s_sqrtB), "s_sqrtB_err": float(s_sqrtB_err)}


def compute_sqrtDS_over_S(data_sig, data_bckg, wp1, wp2, idWP_1, idWP_2, region):
    mask_sig = (
        data_sig["Signal_Fit"]
        & data_sig[region]
        & data_sig[f"mu1_{idWP_1}Id"]
        & data_sig[f"mu2_{idWP_2}Id"]
        & (data_sig["mu1_pfRelIso04_all"] < wp1)
        & (data_sig["mu2_pfRelIso04_all"] < wp2)
    )
    mask_bkg = (
        data_bckg["Signal_Fit"]
        & data_bckg[region]
        & data_bckg[f"mu1_{idWP_1}Id"]
        & data_bckg[f"mu2_{idWP_2}Id"]
        & (data_bckg["mu1_pfRelIso04_all"] < wp1)
        & (data_bckg["mu2_pfRelIso04_all"] < wp2)
    )

    w_sig = data_sig["weight_MC_Lumi_pu"][mask_sig]
    w_bkg = data_bckg["weight_MC_Lumi_pu"][mask_bkg]

    S = w_sig.sum()
    sigma_B = np.sqrt(np.sum(w_bkg**2))

    sqrtDS_over_S = np.sqrt(sigma_B / max(S, 1)) if S > 0 else 0.0
    return float(sqrtDS_over_S)


results = []

for wp1, wp2, idWP_1, idWP_2, region in product(isoWPs, isoWPs, idWPs, idWPs, regions):
    results.append(
        {
            "WP1": wp1,
            "WP2": wp2,
            "Id1": idWP_1,
            "Id2": idWP_2,
            "region": region,
            "signal": compute_eff(sig, wp1, wp2, idWP_1, idWP_2, region),
            "background": compute_eff(bkg, wp1, wp2, idWP_1, idWP_2, region),
            "s_sqrtB": compute_ssqrtb(sig, bkg, wp1, wp2, idWP_1, idWP_2, region),
            "sqrtDS_over_S": compute_sqrtDS_over_S(
                sig, bkg, wp1, wp2, idWP_1, idWP_2, region
            ),
        }
    )

# -------------------------
# Salvataggio
# -------------------------
json.dump(
    results,
    open(
        f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/json/efficienciesTTbar_{args.year}.json",
        "w",
    ),
    indent=2,
)
print(
    f"Efficienze salvate in /afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/json/efficienciesTTbar_{args.year}.json"
)

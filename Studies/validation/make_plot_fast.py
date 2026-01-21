#!/usr/bin/env python3
import uproot
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse

hep.style.use("CMS")

# ----------------------------------------
# Args
# ----------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
args = parser.parse_args()


# ----------------------------------------
# Helpers ISO
# ----------------------------------------
def right_integral_eff(arr, weights, wp):
    """
    Efficienza cumulativa: arr < wp
    """
    if wp is None:
        return None
    tot = np.sum(weights)
    if tot == 0:
        return 0.0
    passed = np.sum(weights[arr < wp])
    return passed / tot


isoWPs_dict = {0.25: "L", 0.2: "M", 0.15: "T"}

# Lumi
period_dict = {"2022": "7.98", "2022EE": "26.67", "2023": "18.06", "2023BPix": "9.69"}
lumi = period_dict.get(args.year, "N/A")


# ----------------------------------------
# Load
# ----------------------------------------
def load(fname):
    f = uproot.open(fname)["Mini"]
    return {k: f[k].array(library="np") for k in f.keys()}


sig = load(
    f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/mini_signal_{args.year}.root"
)
bkg = load(
    f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/mini_bkgFake_{args.year}.root"
)

regions = ["baseline_noID_Iso", "VBF_JetVeto_noID_Iso", "ggH_noID_Iso"]
idWPs = ["loose", "medium", "tight"]


# ----------------------------------------
# Helpers plotting
# ----------------------------------------
def cms_save(fig, out, pre_folder, year):
    pre_path = f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/plots_WithFake_{pre_folder}/Run3_{year}"
    os.makedirs(pre_path, exist_ok=True)
    hep.cms.label("Preliminary", data=True, lumi=lumi)
    fig.savefig(f"{pre_path}/{out}", dpi=350, bbox_inches="tight")
    print(f"fig saved in {pre_path}/{out}")
    plt.close()


# ----------------------------------------
# ISO distributions
# ----------------------------------------
isoWPs = [0.25, 0.20, 0.15]

muons = {
    "mu1": {
        "iso": "mu1_pfRelIso04_all",
        "trg": "mu1_isTrigger",
        "id": lambda wp: f"mu1_{wp}Id",
    },
    "mu2": {
        "iso": "mu2_pfRelIso04_all",
        "trg": "mu2_isTrigger",
        "id": lambda wp: f"mu2_{wp}Id",
    },
}


def plot_iso(
    region,
    data,
    label,
    year,
    overlay_muons=False,
    scan_IDwps=True,
    scan_wps=True,
    draw_cut_lines=True,
):
    for IDwp1 in idWPs if scan_IDwps else ["loose"]:
        for IDwp2 in idWPs if scan_IDwps else ["loose"]:
            for wp1 in isoWPs if scan_wps else [0.25]:
                for wp2 in isoWPs if scan_wps else [0.25]:
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plotted = False
                    for mu_name, mu in muons.items():
                        other = "mu2" if mu_name == "mu1" else "mu1"
                        mask = (
                            data[region]
                            & data[mu["trg"]]
                            & data[mu["id"](IDwp1 if mu_name == "mu1" else IDwp2)]
                            & data[
                                muons[other]["id"](IDwp2 if mu_name == "mu1" else IDwp1)
                            ]
                        )
                        arr = data[mu["iso"]][mask]
                        weights = data["weight_MC_Lumi_pu"][mask]
                        if len(arr) == 0 or weights.sum() == 0:
                            continue
                        wp_cut = wp1 if mu_name == "mu1" else wp2
                        eff_tight = right_integral_eff(arr, weights, 0.15)
                        eff_medium = right_integral_eff(arr, weights, 0.20)
                        eff_loose = right_integral_eff(arr, weights, 0.25)
                        label_eff = f"{mu_name}, {IDwp1} ID\n ε(Loose Iso)={eff_loose:.3f}\n ε(Medium Iso)={eff_medium:.3f}\n ε(Tight Iso)={eff_tight:.3f}"
                        ax.hist(
                            arr,
                            bins=50,
                            weights=weights,
                            histtype="step",
                            linewidth=2,
                            label=label_eff,
                        )
                        plotted = True
                        if not overlay_muons:
                            break
                    if not plotted:
                        plt.close()
                        continue

                    if draw_cut_lines:
                        ax.axvline(
                            0.25,
                            linestyle="--",
                            color="red",
                            label="Loose Iso",
                            linewidth=2,
                        )
                        ax.axvline(
                            0.2,
                            linestyle="--",
                            color="blue",
                            label="Medium Iso",
                            linewidth=2,
                        )
                        ax.axvline(
                            0.15,
                            linestyle="--",
                            color="green",
                            label="Tight Iso",
                            linewidth=2,
                        )
                        # if wp2 is not None: ax.axvline(wp2, linestyle="--", color="blue", linewidth=2)

                    ax.set_xlabel("Muon pfRelIso")
                    ax.set_ylabel("Weighted events")
                    ax.grid(True)
                    ax.legend()
                    cms_save(
                        fig,
                        f"iso_{label}_{region}_{args.year}_ID1{IDwp1}_ID2{IDwp2}_overlay_wp1{isoWPs_dict[wp1]}_wp2{isoWPs_dict[wp2]}.png",
                        "iso",
                        year,
                    )
                    ax.set_yscale("log")
                    cms_save(
                        fig,
                        f"iso_{label}_{region}_{args.year}_ID1{IDwp1}_ID2{IDwp2}_overlay_wp1{isoWPs_dict[wp1]}_wp2{isoWPs_dict[wp2]}_log.png",
                        "iso_log",
                        year,
                    )


# ----------------------------------------
# ID flow
# ----------------------------------------
def plot_id_flow_mu1_mu2(region, data, label, year, isoWP1=0.20, isoWP2=0.20):
    steps, effs = [], []
    id_levels = ["loose", "medium", "tight"]
    for base1 in ["loose", "medium"]:
        for base2 in ["loose", "medium"]:
            base_mask = (
                data[f"mu1_{base1}Id"]
                & data[f"mu2_{base2}Id"]
                & (data["mu1_pfRelIso04_all"] < isoWP1)
                & (data["mu2_pfRelIso04_all"] < isoWP2)
                & data[region]
            )
            den = base_mask.sum()
            if den == 0:
                continue
            for targ1 in ["medium", "tight"]:
                for targ2 in ["medium", "tight"]:
                    if id_levels.index(targ1) <= id_levels.index(base1):
                        continue
                    if id_levels.index(targ2) <= id_levels.index(base2):
                        continue
                    num_mask = (
                        data[f"mu1_{targ1}Id"] & data[f"mu2_{targ2}Id"] & base_mask
                    )
                    # print(targ1,targ2)
                    # print(den)
                    # print(num_mask.sum())
                    # print(num_mask.sum() / den)
                    effs.append(num_mask.sum() / den)
                    steps.append(
                        f"({base1[0].upper()},{base2[0].upper()}) → ({targ1[0].upper()},{targ2[0].upper()})"
                    )

    if not steps:
        print(f"[WARNING] No valid entries for {region}")
        return

    fig, ax = plt.subplots(figsize=(30, 15))
    ax.plot(steps, effs, marker="o", linestyle="-")
    ax.set_ylabel("Efficiency")
    ax.set_ylim(0.8, 1.05)
    ax.grid(True)
    cms_save(
        fig,
        f"idflow_mu1mu2_{label}_{region}_{args.year}_iso1{isoWPs_dict[isoWP1]}_iso2{isoWPs_dict[isoWP2]}.png",
        "WPs",
        year,
    )


# ----------------------------------------
# MAIN LOOP
# ----------------------------------------
for region in regions:
    print(f"\n=== Region:", region)
    plot_iso(
        region,
        sig,
        "signal",
        args.year,
        overlay_muons=True,
        scan_IDwps=True,
        scan_wps=False,
        draw_cut_lines=True,
    )
    plot_iso(
        region,
        bkg,
        "background",
        args.year,
        overlay_muons=True,
        scan_IDwps=True,
        scan_wps=False,
        draw_cut_lines=True,
    )
    for muIso1 in isoWPs:
        for muIso2 in isoWPs:
            plot_id_flow_mu1_mu2(
                region, sig, "signal", args.year, isoWP1=muIso1, isoWP2=muIso2
            )
            plot_id_flow_mu1_mu2(
                region, bkg, "background", args.year, isoWP1=muIso1, isoWP2=muIso2
            )

print("\nDone.")

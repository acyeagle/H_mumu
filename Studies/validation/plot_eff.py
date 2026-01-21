#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from collections import defaultdict
import argparse
import os

hep.style.use("CMS")
parser = argparse.ArgumentParser()
parser.add_argument("--year", required=True)
args = parser.parse_args()

try:
    with open(
        f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/json/efficienciesWithFake_{args.year}.json"
    ) as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Errore: Il file per l'anno {args.year} non è stato trovato.")
    exit()
except json.JSONDecodeError:
    print(f"Errore: Impossibile decodificare il file JSON per l'anno {args.year}.")
    exit()

period_dict = {
    "2022": "7.9804",
    "2022EE": "26.6717",
    "2023": "18.063",
    "2023BPix": "9.693",
}
lumi = period_dict.get(args.year, "N/A")

regions = ["baseline_noID_Iso", "VBF_JetVeto_noID_Iso", "ggH_noID_Iso"]

region_dict = defaultdict(list)
for entry in data:
    region_dict[entry["region"]].append(entry)


def compact_label(entry):
    id_dict = {"loose": "L", "medium": "M", "tight": "T"}
    iso_dict = {0.25: "L", 0.2: "M", 0.15: "T"}

    mu1_id = entry.get("Id1", "")
    mu2_id = entry.get("Id2", "")

    wp1 = entry.get("WP1")
    wp2 = entry.get("WP2")

    mu1_id_label = id_dict.get(mu1_id, mu1_id)
    mu2_id_label = id_dict.get(mu2_id, mu2_id)
    iso1_label = iso_dict.get(wp1, str(wp1))
    iso2_label = iso_dict.get(wp2, str(wp2))

    # return f"mu1{mu1_id_label},{iso1_label}, mu2{mu2_id_label},{iso2_label}"
    return f"mu1{mu1_id_label},{iso1_label}\nmu2{mu2_id_label},{iso2_label}"


def plot_region(region_entries, region_name, lumi):

    region_entries = sorted(
        region_entries,
        key=lambda e: (
            e.get("Id1", ""),
            e.get("Id2", ""),
            float(e.get("WP1", 0)),
            float(e.get("WP2", 0)),
        ),
    )

    x = np.arange(len(region_entries))
    labels = [compact_label(e) for e in region_entries]

    s_sqrtB = np.array([e["s_sqrtB"]["s_sqrtB"] for e in region_entries])
    s_sqrtB_e = np.array([e["s_sqrtB"].get("s_sqrtB_err", 0.0) for e in region_entries])

    eff_sig = np.array([e["signal"]["eff"] for e in region_entries])
    eff_sig_e = np.array([e["signal"].get("err", 0.0) for e in region_entries])

    eff_bck = np.array([e["background"]["eff"] for e in region_entries])
    eff_bck_e = np.array([e["background"].get("err", 0.0) for e in region_entries])

    def make_plot(y, yerr, ylabel, outname):
        fig, ax = plt.subplots(figsize=(80, 15))
        ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=6, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=15)
        ax.set_ylabel(ylabel)

        if y.size > 0:
            y_min = np.min(y[y > 0]) if np.any(y > 0) else 0.0
            y_max = np.max(y)
            if y_max > 0:
                ax.set_ylim(y_min * 0.95, y_max * 1.05)
            else:
                ax.set_ylim(0, 1.0)

        ax.grid()
        hep.cms.label("Preliminary", data=True, lumi=lumi)
        plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.25)
        plt.savefig(outname, dpi=300)
        print(f"file salvato in {outname}")
        plt.close()

    pre_path_common = f"/afs/cern.ch/work/v/vdamante/H_mumu/Studies/validation/stuff/plots_effs_Fakes/Run3_{args.year}"
    os.makedirs(pre_path_common, exist_ok=True)
    make_plot(
        s_sqrtB,
        s_sqrtB_e,
        ylabel="S/√B",
        outname=f"{pre_path_common}/s_sqrtB_{region_name}.png",
    )

    make_plot(
        eff_sig,
        eff_sig_e,
        ylabel="Signal efficiency",
        outname=f"{pre_path_common}/eff_signal_{region_name}.png",
    )

    make_plot(
        eff_bck,
        eff_bck_e,
        ylabel="Background efficiency",
        outname=f"{pre_path_common}/eff_background_{region_name}.png",
    )

    sqrtDS_over_S = np.array([e.get("sqrtDS_over_S", 0.0) for e in region_entries])
    make_plot(
        sqrtDS_over_S,
        None,  # non serve yerr per ora
        ylabel="sqrt(ΔS/S)",
        outname=f"{pre_path_common}/sqrtDS_over_S_{region_name}.png",
    )


for reg in regions:
    if reg in region_dict:
        plot_region(region_dict[reg], reg, lumi)

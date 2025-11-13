import os

import ROOT as root
from tqdm import tqdm

base_dir = os.getcwd()
os.chdir("/eos/user/a/ayeagle/H_mumu/root_files/v3_parse1/")

f = root.TFile("Run3_2022/GluGluHto2Mu.root")
tree = f.Get("Events")


def check_event(event):
    candidate_indices = []
    for i, (pt, eta, phi) in enumerate(zip(event.Jet_pt, event.Jet_eta, event.Jet_phi)):
        if pt < 25:
            continue
        if (abs(eta) < 2.5) or (abs(eta) > 3.0) or (pt > 50):
            candidate_indices.append(i)
    return candidate_indices


count = 0
for event in tqdm(tree, total=tree.GetEntries()):
    result = check_event(event)
    if result:
        count += 1

print("Final count:", count)

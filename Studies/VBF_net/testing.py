import os
from pprint import pprint

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import JetDataset

mplhep.style.use(mplhep.styles.CMS)


class Tester:
    """
    Runs inference on the provided data using the provided model.
    Saves inference to self.testing_df, then produces plots.
    All of the nice output plots are defined here.
    """

    def __init__(
        self,
        testing_df,
        device,
        max_jets,
        n_bins=20,
        **kwargs,
    ):
        # Set self attrs
        self.testing_df = testing_df
        self.device = device
        self.max_jets = max_jets
        self.n_bins = n_bins
        # Other on-the-fly ones
        self.hist_range = (0, 1)
        self.batch_size=1000 

    def collate_fn(self, batch):
        x, y = zip(*batch)
        x = torch.tensor(np.stack(x), dtype=torch.double, device=self.device)
        #y = torch.tensor(np.stack(y), dtype=torch.double, device=self.device)
        return x, y

    def _make_dataloader(self, data):
        return DataLoader(
            data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=2,
        )

    def test(self, model, testing_idx):
        """
        Run the inference, save to testing_df
        """
        #print("Testing indicies:", testing_idx)
        if testing_idx is None:
            selected = self.testing_df
        else:
            selected = self.testing_df.loc[testing_idx]
        data = JetDataset(selected, None, self.max_jets, for_inference=True)
        data = self._make_dataloader(data)
        model.eval()
        with torch.no_grad():
            print("Running testing...")
            for x_data, indices in tqdm(data):
                #print("\tBatch indices:", indices)
                outputs = model(x_data)
                outputs = outputs.cpu().numpy()
                self.testing_df.loc[indices, "NN_Output"] = outputs

    ### Calculations of metrics and other ###

    def get_roc_auc(self):
        """
        Area under curve for ROC
        """
        df = self.testing_df
        fpr, tpr, _ = roc_curve(df.Label, df.NN_Output, sample_weight=df.Class_Weight)
        score = np.trapz(x=tpr, y=fpr)
        return score

    @staticmethod
    def s2overb(signal_bins, background_bins):
        """
        Takes two np arrays. Precompute a histogram and pass the bins over here.
        """
        x = (signal_bins) / np.sqrt(background_bins + signal_bins)
        return np.sqrt(np.sum(x**2))

    ### PLOTTING ###

    def make_hist(self, weight=True, log=True, norm=False, show=False):
        """
        Saves a histo of all signal vs all background
        """
        plt.clf()
        output_name = "model_hist"
        results = self.testing_df
        signal = results[results.Label == 1]
        background = results[results.Label == 0]
        if norm:
            w = sum(background.Class_Weight) / sum(signal.Class_Weight)
            output_name += "_normed"
        else:
            w = 1
        if weight:
            h1 = np.histogram(
                signal.NN_Output,
                range=self.hist_range,
                bins=self.n_bins,
                weights=signal.Class_Weight * w,
            )
            h2 = np.histogram(
                background.NN_Output,
                range=self.hist_range,
                bins=self.n_bins,
                weights=background.Class_Weight,
            )
        else:
            h1 = np.histogram(signal.NN_Output, range=self.hist_range, bins=self.n_bins)
            h2 = np.histogram(
                background.NN_Output, range=self.hist_range, bins=self.n_bins
            )
        plt.stairs(*h2, label="Background", color="tab:orange")
        plt.stairs(*h1, label="Signal", color="tab:blue")
        # Set plot parameters based on boolean options
        if log:
            plt.yscale("log")
            output_name += "_log"
        else:
            output_name += "_lin"
        if weight:
            plt.ylabel("Weight")
            output_name += "_weighted"
        else:
            plt.ylabel("Events")
        # Set rest of plot and save/show
        plt.xlabel("Network output")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

    def make_multihist(self, weight=True, log=False, show=False):
        """
        Saves a histo with the different processes drawn independently
        """
        # Init plot
        plt.clf()
        df = self.testing_df
        # Add individual hist curves
        for p in sorted(pd.unique(df.process)):
            selected = df[df.process == p]
            if weight:
                h = np.histogram(
                    selected.NN_Output,
                    weights=selected.Class_Weight,
                    range=self.hist_range,
                    bins=self.n_bins,
                )
            else:
                h = np.histogram(
                    selected.NN_Output, range=self.hist_range, bins=self.n_bins
                )
            plt.stairs(*h, label=p)
        # Plot config from boolean parameters
        output_name = "multihist"
        if log:
            plt.yscale("log")
        if weight:
            output_name += "_weighted"
            plt.ylabel("Weight")
        else:
            plt.ylabel("Events")
        # Finish plot config
        plt.xlabel("Network output")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        mplhep.cms.label(com=13.6, lumi=62.4)
        # Out (save or show)
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

    def make_roc_plot(self, log=True, hist_points=False, show=False):
        """
        Plot ROC and adds AUC calculation to plot
        """
        plt.clf()
        df = self.testing_df
        output_name = "roc"
        df = self.testing_df
        # Calc curve from sklearn
        fpr, tpr, _ = roc_curve(df.Label, df.NN_Output, sample_weight=df.Class_Weight)
        plt.plot(tpr, fpr, label=r"$DNN$")
        # Do a by hand calc (set hist_points to True to check the sklearn calc)
        if hist_points:
            sig_eff, bkg_eff = self.by_hand_roc_calc()
            plt.scatter(sig_eff, bkg_eff, label="hists", color="tab:orange")
            output_name += "_whp"
        # Add 45 deg
        a = np.linspace(0, 1, 1000)
        plt.plot(a, a, color="black", linestyle="dashed", label="45°")
        # Add score
        auc = self.get_roc_auc()
        x = 0.6
        if log:
            y = 3e-4
        else:
            y = 0
        text = f"1 - AUC = {round(1-auc, 3)}"
        plt.text(x, y, text)
        # Format and go!
        plt.xlabel(r"$\epsilon_{sig}$")
        plt.ylabel(r"$\epsilon_{bkg}$")
        # mplhep.cms.label()
        plt.grid()
        plt.xlim(0, 1)
        if log:
            plt.yscale("log")
            plt.ylim(1e-4, 1)
            output_name += "_log"
        plt.legend(loc="upper left")
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

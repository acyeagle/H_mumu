import os

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve
from pprint import pprint
from tqdm import tqdm

mplhep.style.use(mplhep.styles.CMS)


class Tester:
    """
    Runs inference on the provided data using the provided model.
    Saves inference to self.testing_df, then produces plots.
    All of the nice output plots are defined here.
    """

    def __init__(
        self,
        testing_data,
        testing_df,
        n_bins=30,
        device=None,
        **kwargs,
    ):
        # Set self attrs
        self.testing_df = testing_df
        self.device = device
        self.n_bins = n_bins
        # Other on-the-fly ones
        self.hist_range = (0, 1)

        # Just keep and convert the x_data.
        # We'll run inference on this only then put back into self.testing_df
        (x_data, _), _ = testing_data
        if self.device is None:
            self.x_data = torch.tensor(x_data, device=self.device)
        else:
            self.x_data = torch.tensor(x_data, device=self.device, dtype=torch.double)
        # Define a mapping, so it is consistent across plots


    def test(self, model):
        """
        Run the inference, save to testing_df
        """
        outputs = []
        model.eval()
        with torch.no_grad():
            print("Running testing...")
            total = len(self.x_data)
            outputs = model(self.x_data)
        outputs = outputs.cpu().numpy()
        self.testing_df["NN_Output"] = outputs

    ### Calculations of metrics and other ###

    def get_roc_auc(self):
        """
        Area under curve for ROC
        """
        df = self.testing_df
        fpr, tpr, _ = roc_curve(df.Label, df.NN_Output, sample_weight=df.Class_Weight)
        score = np.trapz(x=tpr, y=fpr)
        return score


    ### PLOTTING ###

    def make_hist(self, weight=True, log=False, norm=False, show=False):
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
        mplhep.cms.label(com=13.6, lumi=62.4)
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
        #mplhep.cms.label()
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

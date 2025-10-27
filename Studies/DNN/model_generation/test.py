import os

import matplotlib.pyplot as plt
import mplhep
import numpy as np
import pandas as pd
import ROOT as root
import torch
import uproot
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
from statsmodels.stats.weightstats import DescrStatsW
from pprint import pprint

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
        signal_types,
        classification,
        n_bins=30,
        device=None,
        **kwargs,
    ):
        # Set self attrs
        self.testing_df = testing_df
        self.device = device
        self.classification = classification
        self.signal_types = signal_types
        self.n_bins = n_bins
        # Other on-the-fly ones
        self.hist_range = (0, 1)
        self.processes = sorted(pd.unique(testing_df.process))

        # Just keep and convert the x_data.
        # We'll run inference on this only then put back into self.testing_df
        (x_data, _), _ = testing_data
        if self.device is None:
            self.x_data = torch.tensor(x_data, device=self.device)
        else:
            self.x_data = torch.tensor(x_data, device=self.device, dtype=torch.double)
        # Define a mapping, so it is consistent across plots
        self.color_map = self._set_color_map()

    def _set_color_map(self):
        return {
            "DY": "gold",
            "TT": "blue",
            "VV": "lightslategray",
            "EWK": "magenta",
            "VBFHto2Mu": "green",
            "GluGluHto2Mu": "red",
        }

    def prediction_to_prob(self, subdf):
        """
        Function for converting a a set of multiclass probabilities
        into a single discriminant for plotting
        """
        classes = [x.replace("Prob_", "") for x in subdf.columns]
        predictions = subdf.apply(np.argmax, axis=1).apply(lambda x: classes[x])
        bkg_processes = [x for x in classes if x not in self.signal_types]
        # Calc an output discriminator (from the Run2 paper)
        final_prob = np.zeros(len(subdf))
        for p in self.signal_types:
            final_prob += subdf[f"Prob_{p}"]
        for p in bkg_processes:
            final_prob -= subdf[f"Prob_{p}"]
        # Rescale so discrim is in [0,1]
        final_prob += len(bkg_processes)
        final_prob /= len(classes)
        # maxima = subdf.apply(np.max, axis=1).values.copy()
        # mask = np.isin(predictions, self.signal_types)
        # maxima[~mask] = 1 - maxima[~mask]
        return predictions, final_prob

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
        if self.classification == "binary":
            outputs = outputs.cpu().numpy()
            self.testing_df["NN_Output"] = outputs
        else:
            cols = [f"Prob_{x}" for x in self.processes]
            self.testing_df[cols] = outputs.cpu().numpy()
            predictions, probs = self.prediction_to_prob(self.testing_df[cols])
            self.testing_df["Prediction"] = predictions
            self.testing_df["NN_Output"] = probs

    ### Calculations of metrics and other ###

    def get_roc_auc(self):
        """
        Area under curve for ROC
        """
        df = self.testing_df
        fpr, tpr, _ = roc_curve(df.Label, df.NN_Output, sample_weight=df.Class_Weight)
        score = np.trapz(x=tpr, y=fpr)
        return score

    def by_hand_roc_calc(self):
        """
        My quick implementation, just a check for the sklearn calc
        """
        df = self.testing_df
        sig = df[df.Label == 1]
        bkg = df[df.Label == 0]
        sig_hist, sig_bins = np.histogram(
            sig.NN_Output,
            weights=sig.Class_Weight,
            range=self.hist_range,
            bins=self.n_bins,
        )
        bkg_hist, bkg_bins = np.histogram(
            bkg.NN_Output,
            weights=bkg.Class_Weight,
            range=self.hist_range,
            bins=self.n_bins,
        )
        sig_eff = 1 - (np.cumsum(sig_hist) / np.sum(sig_hist))
        bkg_eff = 1 - (np.cumsum(bkg_hist) / np.sum(bkg_hist))
        return sig_eff, bkg_eff

    # def _calc_transformed_hist(self):
    #     """
    #     Counts the populations for the DNN' plots from AN2019_205, fig 14
    #     """
    #     df = self.testing_df
    #     # Calculate bin edges for percentiles
    #     signal = df[df.Label == 1]
    #     q = np.linspace(0, 100, self.n_bins + 1)
    #     bin_edges = np.percentile(signal.NN_Output * signal.Class_Weight, q)
    #     # Calculate the bin populations for each process
    #     counts_lookup = {}
    #     for p in pd.unique(df.process):
    #         selected = df[df.process == p]
    #         counts, _ = np.histogram(
    #             selected.NN_Output, weights=selected.Class_Weight, bins=bin_edges
    #         )
    #         counts_lookup[p] = counts
    #     return bin_edges, counts_lookup

    def _calc_transformed_hist(self):
        """
        Counts the populations for the DNN' plots from AN2019_205, fig 14
        """
        df = self.testing_df
        # Calculate bin edges for percentiles
        signal = df[df.process == 'VBFHto2Mu']
        wq = DescrStatsW(data=signal.NN_Output, weights=signal.Class_Weight)
        p = np.linspace(0, 1, self.n_bins + 1)
        bin_edges = wq.quantile(p, return_pandas=False)
        # Calculate the bin populations for each process
        counts_lookup = {}
        for p in pd.unique(df.process):
            selected = df[df.process == p]
            counts, _ = np.histogram(
                selected.NN_Output, weights=selected.Class_Weight, bins=bin_edges
            )
            counts_lookup[p] = counts
        print("Bin edges:", bin_edges)
        pprint(counts_lookup)
        return bin_edges, counts_lookup

    @staticmethod
    def s2overb(signal_bins, background_bins):
        """
        Takes two np arrays. Precompute a histogram and pass the bins over here.
        """
        x = (signal_bins) / np.sqrt(background_bins + signal_bins)
        return np.sqrt(np.sum(x**2))

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
        mplhep.cms.label(com=13.6)
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
            plt.stairs(*h, label=p, color=self.color_map[p])
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
        mplhep.cms.label(com=13.6)
        # Out (save or show)
        if show:
            plt.show()
        else:
            plt.savefig(output_name + ".svg", bbox_inches="tight")

    def make_stackplot(self, log=True, show=False):
        """
        Regularly binned stackplot (x-axis is NN output 0 to 1)
        """
        output_name = "stackplot"
        plt.clf()
        df = self.testing_df
        # Only stack background processes
        # Yeah yeah hardcoding sig is bad, I know.
        sig = ["VBFHto2Mu", "GluGluHto2Mu"]
        bkg = [x for x in pd.unique(df.process) if x not in sig]

        # Make sure sorted for stacking
        def size(p):
            selected = df[df.process == p]
            return selected.Class_Weight.sum()

        # Do the stack part plot
        bkg = sorted(bkg, key=lambda x: size(x))
        weights = []
        values = []
        for p in bkg:
            selected = df[df.process == p]
            weights.append(selected.Class_Weight.values)
            values.append(selected.NN_Output.values)
        colors = [self.color_map[x] for x in bkg]
        plt.hist(
            values,
            weights=weights,
            bins=self.n_bins,
            range=self.hist_range,
            stacked=True,
            label=bkg,
            color=colors,
            alpha=0.9,
        )
        # Now add on the signal curves
        for p in sig:
            selected = df[df.process == p]
            h = np.histogram(
                selected.NN_Output,
                weights=selected.Class_Weight,
                bins=self.n_bins,
                range=self.hist_range,
            )
            plt.stairs(*h, label=p, color=self.color_map[p])
        # Format
        if log:
            plt.yscale("log")
            output_name += "_log"
        plt.xlabel("Network output")
        plt.ylabel("Events")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        mplhep.cms.label(com=13.6)
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
        mplhep.cms.label(com=13.6)
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

    def make_transformed_stackplot(
        self, log=True, show=False, min_power=-3, top_power=6
    ):
        """
        Replication for the DNN' plots from AN2019_205, fig 14
        """
        plt.clf()
        df = self.testing_df
        output_name = "transformed_stack"
        bin_edges, counts_lookup = self._calc_transformed_hist()

        # HARDCODE WARNING! (Ricardo wants this plot in a specific order)
        stack_proc = ["VV", "TT", "EWK", "DY"]
        other_proc = [x for x in pd.unique(df.process) if x not in stack_proc]

        # Do the stacking
        baseline = np.zeros(len(bin_edges) - 1)
        total = np.zeros(len(baseline))
        x = np.linspace(0, self.n_bins, self.n_bins + 1)
        for p in stack_proc:
            total += counts_lookup[p]
            plt.stairs(
                total, x, label=p, color=self.color_map[p], baseline=baseline, fill=True
            )
            baseline += counts_lookup[p]
        # and add the non-stackers
        for p in other_proc:
            plt.stairs(counts_lookup[p], x, label=p, color=self.color_map[p])

        # Add S/sqrt(B) to plot
        sig_total = np.zeros(len(baseline))
        for p in self.signal_types:
            sig_total += counts_lookup[p]
        sensitivity = self.s2overb(sig_total, total)
        plt.text(1, 1E5, r"$\frac{S}{\sqrt{S + B}} = $" + str(round(sensitivity, 3)))

        # Format
        if log:
            output_name += "_log"
            plt.yscale("log")
        plt.xlabel(r"$DNN^{\prime}$")
        major_ticks = [1 * 10**x for x in range(min_power, top_power + 1)]
        minor_ticks = [
            y * 10**x for y in range(2, 10) for x in range(min_power, top_power - 1)
        ]
        plt.ylim(10**min_power, 10**top_power)
        plt.ylabel("Events")
        plt.yticks(major_ticks)
        plt.yticks(minor_ticks, minor=True)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.legend()
        mplhep.cms.label(com=13.6)
        if show:
            plt.show()
        else:
            output_name += ".svg"
            plt.savefig(output_name, bbox_inches="tight")

    def plot_multiclass_probs(self, log=True, show=False):
        """
        Multiclass mode only!
        Make a figure with a subplot for each process.
        Shows the process population vs rest of population
        as a function of that process discriminant (P_{process})
        """
        basedir = os.getcwd()
        if not os.path.isdir('multiclass_hists'):
            os.mkdir('multiclass_hists')
        os.chdir('multiclass_hists')
        df = self.testing_df
        proc = pd.unique(df.process)
        alpha = 0.8
        for p in proc:
            plt.clf()
            mask = df.process == p
            selected = df[mask]
            other = df[~mask]
            col = f"Prob_{p}"
            plt.hist(selected[col], label=p, weights=selected.Class_Weight, alpha=alpha, bins=self.n_bins, range=self.hist_range)
            plt.hist(other[col], label='Else', weights=other.Class_Weight, alpha=alpha, bins=self.n_bins, range=self.hist_range)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{p}_hist.svg")
        os.chdir(basedir)

    ### Functions for working with Combine

    def make_thist(self):
        """
        Saves a THist to be used with combine.
        Combine datacard expect two hists named:
        "signal" and "background"
        """
        df = self.testing_df
        bin_edges, counts_lookup = self._calc_transformed_hist()
        # Calc sig and bkg totals
        sig = np.zeros(len(bin_edges) - 1)
        bkg = np.zeros(len(bin_edges) - 1)
        for p in pd.unique(df.process):
            x = counts_lookup[p]
            if p in self.signal_types:
                sig += x
            else:
                bkg += x
        # Make the histograms
        sig_hist = root.TH1F("signal", "signal", self.n_bins, 0, self.n_bins)
        bkg_hist = root.TH1F("background", "background", self.n_bins, 0, self.n_bins)
        for i, (s, b) in enumerate(zip(sig, bkg)):
            sig_hist.SetBinContent(i + 1, s)
            bkg_hist.SetBinContent(i + 1, b)
        # Save to a root file
        with uproot.recreate("hists.root") as f:
            f["signal"] = sig_hist
            f["background"] = bkg_hist

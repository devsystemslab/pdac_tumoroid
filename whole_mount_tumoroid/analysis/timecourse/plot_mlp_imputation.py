import sys
sys.path.append('/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen')

import yaml
import pandas as pd
from pathlib import Path
import muon as mu
import numpy as np
import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

def plot_evaluation(df, marker_names):
    n = len(marker_names)
    ncols = min(n, 8)
    nrows = 2 * ((n + ncols - 1) // ncols)  # 2 rows (scatter + residual) per marker row

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.reshape(nrows, ncols)  # ensure 2D even if single row

    metrics = []
    for i, marker in enumerate(marker_names):
        sub = df[df["marker"] == marker]
        pred, target = sub["pred"].values, sub["target"].values

        r2       = r2_score(target, pred)
        pearson  = pearsonr(target, pred)[0]
        spearman = spearmanr(target, pred)[0]
        mae      = mean_absolute_error(target, pred)
        metrics.append({"marker": marker, "R²": r2, "Pearson": pearson, "Spearman": spearman, "MAE": mae})

        ncols = 8
        col = i % ncols
        scatter_row = (i // ncols) * 2
        resid_row   = scatter_row + 1
        
        # --- scatter: predicted vs actual ---
        ax = axes[scatter_row, col]
        ax.scatter(target, pred, alpha=0.3, s=5, rasterized=True)
        lims = [min(target.min(), pred.min()), max(target.max(), pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)  # identity line
        ax.set_xlabel("Measured intensity")
        ax.set_ylabel("Predicted intensity")
        ax.set_title(f"{marker}\nR²={r2:.3f}  ρ={spearman:.3f}")

        # --- residual distribution ---
        ax = axes[resid_row, col]
        residuals = pred - target
        ax.hist(residuals, bins=50, edgecolor="none")
        ax.axvline(0, color="r", linestyle="--")
        ax.set_xlabel("Residual (pred - actual)")
        ax.set_ylabel("Count")
        ax.set_title(f"{marker} residuals\nMAE={mae:.3f}")

    plt.tight_layout()
    #plt.savefig("plots/mlp_evaluation.pdf")
    plt.show()

    metrics_df = pd.DataFrame(metrics).set_index("marker")
    print(metrics_df.round(3))
    return metrics_df

def plot_comparison(mlp_metrics, baseline_metrics, nn_metrics, marker_names):
    metric_names = ["Spearman"]
    fig, ax = plt.subplots(1, len(metric_names), figsize=(10 * len(metric_names), 8))

    colors = {
        "MLP Nuclei":      "steelblue",
        "MLP Neighbors":   "cornflowerblue",
        "Ridge Nuclei":    "coral",
        "Ridge Neighbors": "lightsalmon",
        "kNN Nuclei":   "mediumseagreen",
        "kNN Neighbors":"lightgreen",

    }
    datasets = {
        "MLP Nuclei":      mlp_metrics["nuclei"],
        "MLP Neighbors":   mlp_metrics["neighbors"],
        "Ridge Nuclei":    baseline_metrics["nuclei"],
        "Ridge Neighbors": baseline_metrics["neighbors"],
        "kNN Nuclei":   nn_metrics["nuclei"],
        "kNN Neighbors":nn_metrics["neighbors"],
    }
    n_models    = len(datasets)
    group_width = n_models + 1

    for metric in metric_names:
        sorted_markers = sorted(
            marker_names,
            key=lambda m: datasets["MLP Neighbors"][datasets["MLP Neighbors"]["marker"] == m][metric].mean(),
            reverse=(metric != "MAE")
        )
        n = len(sorted_markers)

        for model_offset, (label, df) in enumerate(datasets.items()):
            positions = np.arange(n) * group_width + model_offset
            ax.boxplot(
                [df[df["marker"] == m][metric].values for m in sorted_markers],
                positions=positions,
                patch_artist=True,
                boxprops=dict(facecolor=colors[label], alpha=0.8),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                widths=0.7,
            )

        #centre_offset = (n_models - 1) / 2
        ax.set_xticks(np.arange(n) * group_width)
        ax.set_xticklabels(sorted_markers, rotation=90, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)

        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in colors.items()])

    plt.tight_layout()
    #plt.savefig("plots/mlp_comparison.pdf")
    plt.show()

def plot_summary(mlp_metrics, baseline_metrics, nn_metrics, marker_names, colors, datasets, jitter_points = False, save_plot = False):
    metric_names = ["Spearman"]


    # Plot 1: one box per model, all markers pooled
    fig, ax = plt.subplots(1, len(metric_names), figsize=(10, 5))

    for metric in metric_names:
        ax.boxplot(
            [df[metric].values for df in datasets.values()],
            labels=list(datasets.keys()),
            patch_artist=True,
            boxprops=dict(alpha=0.8),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            widths=0.6,
        )
        # colour each box
        for patch, color in zip(ax.patches, colors.values()):
            patch.set_facecolor(color)

        if jitter_points:
            # jittered points
            for i, (key, df) in enumerate(datasets.items(), start=1):
                d = df[metric].values
                jitter = np.random.uniform(-0.15, 0.15, size=len(d))
                color = colors[key]
                ax.scatter(i + jitter, d, color=color, alpha=0.6, s=25, zorder=3,
                        edgecolors="white", linewidths=0.4)

        ax.set_ylabel(f"{metric} correlation")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


    plt.tight_layout()
    if save_plot:
        plt.savefig(f"plots_new/mlp_summary_boxplot_jitter{jitter_points}.pdf", dpi=72)
    plt.show()

    # Plot 2: heatmap — models × markers
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(len(marker_names) * 0.7, len(datasets) * 0.5))

        sorted_markers = sorted(
            marker_names,
            key=lambda m: datasets["MLP Neighbors"][datasets["MLP Neighbors"]["marker"] == m][metric].mean(),
            reverse=(metric != "MAE"),
        )

        means = np.zeros((len(datasets), len(sorted_markers)))

        for i, df in enumerate(reversed(datasets.values())):
            for j, marker in enumerate(sorted_markers):
                vals        = df[df["marker"] == marker][metric].values
                means[i, j] = vals.mean()

        cmap = plt.cm.RdBu
        vmin, vmax = means.min(), means.max()

        im = ax.imshow(means, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        ax.set_xticks(range(len(sorted_markers)))
        ax.set_xticklabels(sorted_markers, rotation=90, ha="right", fontsize=8)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(list(reversed(datasets.keys())), fontsize=9)

        plt.colorbar(im, ax=ax, label=metric, shrink=0.5)

        plt.tight_layout()
        if save_plot:
            plt.savefig(f"plots_new/mlp_heatmap_{metric.replace('²', '2')}.pdf", dpi=72)
        plt.show()
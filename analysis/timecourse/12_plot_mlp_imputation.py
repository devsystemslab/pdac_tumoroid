import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


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

        r2 = r2_score(target, pred)
        pearson = pearsonr(target, pred)[0]
        spearman = spearmanr(target, pred)[0]
        mae = mean_absolute_error(target, pred)
        metrics.append(
            {
                "marker": marker,
                "R²": r2,
                "Pearson": pearson,
                "Spearman": spearman,
                "MAE": mae,
            }
        )

        ncols = 8
        col = i % ncols
        scatter_row = (i // ncols) * 2
        resid_row = scatter_row + 1

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
    # plt.savefig("plots/mlp_evaluation.pdf")
    plt.show()

    metrics_df = pd.DataFrame(metrics).set_index("marker")
    print(metrics_df.round(3))
    return metrics_df


def plot_comparison(mlp_metrics, baseline_metrics, nn_metrics, marker_names):
    metric_names = ["Spearman"]
    fig, ax = plt.subplots(1, len(metric_names), figsize=(10 * len(metric_names), 8))

    colors = {
        "MLP Nuclei": "steelblue",
        "MLP Neighbors": "cornflowerblue",
        "Ridge Nuclei": "coral",
        "Ridge Neighbors": "lightsalmon",
        "kNN Nuclei": "mediumseagreen",
        "kNN Neighbors": "lightgreen",
    }
    datasets = {
        "MLP Nuclei": mlp_metrics["nuclei"],
        "MLP Neighbors": mlp_metrics["neighbors"],
        "Ridge Nuclei": baseline_metrics["nuclei"],
        "Ridge Neighbors": baseline_metrics["neighbors"],
        "kNN Nuclei": nn_metrics["nuclei"],
        "kNN Neighbors": nn_metrics["neighbors"],
    }
    n_models = len(datasets)
    group_width = n_models + 1

    for metric in metric_names:
        sorted_markers = sorted(
            marker_names,
            key=lambda m: datasets["MLP Neighbors"][datasets["MLP Neighbors"]["marker"] == m][metric].mean(),
            reverse=(metric != "MAE"),
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

        # centre_offset = (n_models - 1) / 2
        ax.set_xticks(np.arange(n) * group_width)
        ax.set_xticklabels(sorted_markers, rotation=90, ha="right")
        ax.set_title(metric)
        ax.set_ylabel(metric)

        from matplotlib.patches import Patch

        ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in colors.items()])

    plt.tight_layout()
    # plt.savefig("plots/mlp_comparison.pdf")
    plt.show()

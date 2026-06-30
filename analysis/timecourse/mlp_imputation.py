import anndata as ad
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from analysis.timecourse.mlp_training import ConditionalMarkerMLP
from phenocoder.cluster import run_clustering


@torch.no_grad()
def predict_all_markers(
    model: ConditionalMarkerMLP,
    dapi_embeddings: torch.Tensor,
    num_markers: int,
    batch_size: int = 512,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    model.eval().to(device)
    N = dapi_embeddings.shape[0]
    results = torch.zeros(N, num_markers)

    for marker_id in range(num_markers):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = dapi_embeddings[start:end].to(device)
            marker_ids = torch.full((end - start,), marker_id, dtype=torch.long, device=device)
            results[start:end, marker_id] = model(batch, marker_ids).cpu()

    return results


def mlp_impute(mdata, mlp_dir, base_modality, imputations, models, n_comps_pca, resolution, stains):

    adata = mdata[base_modality]

    DAPI_DIM = adata.X.shape[1]
    NUM_MARKERS = len(stains)

    model = ConditionalMarkerMLP(
        dapi_dim=DAPI_DIM,
        num_markers=NUM_MARKERS,
        hidden_dims=[512, 256, 128],
        dropout=0.1,
    )

    for imputation in imputations:
        model_weights_path = f"{mlp_dir}/{models[imputation]}"
        print(f"Loading model {model_weights_path} for {imputation} imputation")

        model.load_state_dict(torch.load(model_weights_path))

        imputed = predict_all_markers(
            model,
            torch.tensor(adata.X, dtype=torch.float32),
            NUM_MARKERS,
            batch_size=128,
        )

        imputed_df = pd.DataFrame(imputed, index=adata.obs_names, columns=stains)

        mdata.mod[f"{base_modality}_{imputation}_imputed"] = ad.AnnData(
            X=imputed_df.values,
            obs=adata.obs.copy(),  # carry over cell metadata
        )
        mdata.mod[f"{base_modality}_{imputation}_imputed"].var_names = stains

        mdata.mod[f"{base_modality}_{imputation}_imputed"] = run_clustering(
            mdata[f"{base_modality}_{imputation}_imputed"],
            n_comps=n_comps_pca,
            resolution=resolution,
            harmony=False,
            use_gpu=False,
            subsampling=True,
            frac=0.1,
        )

    return mdata


def plot_summary(
    mlp_metrics,
    baseline_metrics,
    nn_metrics,
    marker_names,
    colors,
    datasets,
    jitter_points=False,
    save_plot=False,
):
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
                ax.scatter(
                    i + jitter,
                    d,
                    color=color,
                    alpha=0.6,
                    s=25,
                    zorder=3,
                    edgecolors="white",
                    linewidths=0.4,
                )

        ax.set_ylabel(f"{metric} correlation")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

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
                vals = df[df["marker"] == marker][metric].values
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

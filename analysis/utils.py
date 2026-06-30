import io as io_builtin

import anndata as ad
import bbknn
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.cluster import hierarchy as sch
from skimage.util import dtype_limits
from tqdm import tqdm

from phenocoder.spatial import get_chulls_connected_components


def add_scalebar(img: np.ndarray, width: int, height: int, x: int, y: int) -> np.ndarray:
    """
    Add a scale bar (filled rectangle) to an RGB image.
    :param img:
    :param width:
    :param height:
    :param x:
    :param y:
    :return:
    """
    img = img.copy()
    max_val = dtype_limits(img)[1]
    img[y : y + height, x : x + width, :] = max_val
    return img


def order_genes(adata, preselected_genes=None):
    if preselected_genes is not None:
        adata = adata[:, preselected_genes]
    Z = sch.linkage(adata.X.T, method="ward")
    dendrogram = sch.dendrogram(Z, no_plot=True)
    ordered_genes = [adata.var_names[i] for i in dendrogram["leaves"]]
    return ordered_genes


def plot_dotplot(adata, cycle, dir_screen, remove_cluster: list = None):
    sc.set_figure_params(figsize=(20, 20))
    sc.settings.figdir = f"{dir_screen}/plots"
    if remove_cluster is not None:
        adata = adata[~adata.obs["leiden_phenocoder"].isin(remove_cluster)]
    dp = sc.pl.dotplot(
        adata,
        var_names=order_genes(adata),
        groupby="leiden_phenocoder",
        dendrogram=True,
        return_fig=True,
    )
    dp.add_totals(color="grey").style(dot_edge_color="black", dot_edge_lw=0.5, cmap="Greys", dot_max=0.5, dot_min=0.1)
    dp.savefig(f"{dir_screen}/plots/dotplot_cycle_{cycle}.pdf")
    plt.close("all")


def plot_umap(adata, cycle, dir_screen, size=10, color="leiden"):
    n_colors = adata.obs[color].nunique()
    tab20 = sns.color_palette("tab20", n_colors=n_colors)
    adata.uns["leiden_colors"] = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in tab20]
    sc._settings.settings._vector_friendly = True
    fig = sc.pl.umap(adata, color=color, return_fig=True, show=False, s=size)
    plt.savefig(f"{dir_screen}/plots/umap_leiden_cycle_{cycle}.pdf", format="pdf", dpi=300)
    plt.close("all")


def plot_paga(adata):
    n_colors = adata.obs["leiden"].nunique()
    tab20 = sns.color_palette("tab20", n_colors=n_colors)
    adata.uns["leiden_colors"] = [f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in tab20]
    sc.tl.paga(adata, groups="leiden")
    sc.pl.paga(adata, color="leiden", node_size_scale=10, edge_width_scale=2)


def add_black_border(img, width):
    """
    Add black border to image
    :param img:
    :param width:
    :return:
    """
    return np.pad(
        img,
        ((width, width), (width, width), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def plot_organoid(
    id,
    adata,
    project_3d=False,
    edgecolors=None,
    add_legend=False,
    cut_open=False,
    bg_color="black",
    legend_color="white",
    leiden_colors=None,
):
    """
    Plot organoid
    :param id:
    :param adata:
    :param project_3d:
    :param edgecolors:
    :param add_legend:
    :return:
    """
    cmap = ListedColormap(sns.color_palette("tab20", n_colors=adata.obs["leiden"].nunique()))
    # specify color look up per leiden and add to adata.obs using cmap
    if "id" not in adata.obs.columns:
        adata.obs["id"] = adata.obs["well_id"].astype(str) + "_" + adata.obs["plate_id"].astype(str)

    adata = adata[adata.obs["id"] == id]
    if cut_open:
        center_centroid_0 = 3814 / 2
        center_centroid_1 = 3814 / 2
        adata = adata[~((adata.obs["centroid-0"] < center_centroid_0) & (adata.obs["centroid-1"] > center_centroid_1))]
    np.random.seed(0)
    random_indices = np.random.permutation(list(range(adata.shape[0])))
    adata = adata[random_indices, :]
    if leiden_colors is None:
        list_colors = [cmap.colors[i] for i in adata.obs["leiden"].astype(int)]
    else:
        list_colors = [leiden_colors[i] for i in adata.obs["leiden"].astype(int)]
    x_lim, y_lim = (0, 3814), (0, 3814)
    fig = plt.figure(figsize=(5, 5))
    # set background to black
    fig.patch.set_facecolor(bg_color)
    if project_3d:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.set_aspect("equal", "box")
        ax.scatter(
            adata.obs["centroid-1"],
            adata.obs["centroid-0"],
            adata.obs["z"],
            c=list_colors,
            s=20,
            alpha=0.75,
            edgecolors=edgecolors,
            linewidths=0.5,
        )
        # no axis labels
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        ax.axes.get_zaxis().set_ticklabels([])
        ax.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
        ax.set_ylim(ymin=y_lim[0], ymax=y_lim[1])
        ax.set_facecolor(bg_color)
        if cut_open:
            # add line on x-y plane which show the cut off quadrant:
            ax.plot(
                [center_centroid_1, center_centroid_1],
                [0, 3814],
                color="black",
                linestyle="--",
                linewidth=1,
            )
            ax.plot(
                [0, 3814],
                [center_centroid_0, center_centroid_0],
                color="black",
                linestyle="--",
                linewidth=1,
            )
            # center y vertical line into z direction
            ax.plot(
                [center_centroid_1, center_centroid_1],
                [center_centroid_0, center_centroid_0],
                [0, adata.obs["z"].max()],
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=10,
            )
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_aspect("equal", "box")
        ax.scatter(
            adata.obs["centroid-1"],
            adata.obs["centroid-0"],
            c=list_colors,
            s=20,
            edgecolors=edgecolors,
            linewidths=0.5,
        )
        # set axis limits
        ax.set_xlim(xmin=x_lim[0], xmax=x_lim[1])
        ax.set_ylim(ymin=y_lim[0], ymax=y_lim[1])
        # reverse y-axis
        ax.yaxis.set_inverted(True)
        # no axis labels
        ax.axes.get_xaxis().set_ticklabels([])
        ax.axes.get_yaxis().set_ticklabels([])
        # no axis ticks
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])
        # no frame
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        # black background
        ax.set_facecolor(bg_color)
    if add_legend:
        handles = [
            mpatches.Patch(color=cmap.colors[i], label=f"Cluster {i}") for i in range(adata.obs["leiden"].nunique())
        ]
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(1, 1),
            facecolor=legend_color,
            edgecolor=legend_color,
        )
    # tight layout
    plt.tight_layout()
    # reduce white margin around plot
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    # Convert the canvas to a raw RGB buffer
    # Create a bytes buffer to save the plot
    buf = io_builtin.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # Open the PNG image from the buffer and convert it to a NumPy array
    image = np.array(Image.open(buf))
    plt.close(fig)
    return image


def run_umap_gpu(mdata, rep="X"):
    """
    Run UMAP on GPU for all modalities in mdata
    :param mdata:
    :param rep:
    :return:
    """
    for mod in tqdm(mdata.mod_names, desc="Running UMAP"):
        adata = mdata[mod].copy()
        rsc.get.anndata_to_GPU(adata)
        rsc.pp.neighbors(adata, use_rep=rep)
        rsc.tl.umap(adata, n_components=2)
        rsc.get.anndata_to_CPU(adata)
        mdata.mod[mod] = adata.copy()
    mdata.update()
    return mdata


def add_chull_stats_to_mdata_org(
    mdata,
    df_chulls_agg,
    batch_correction=True,
    mod="phenocoder_combined",
    select_variable_features=True,
):
    """
    Add chull data to X of selected mod in mdata.
    :param mdata:
    :param mod:
    :param df_chulls_agg:
    :param batch_correction:
    :param select_variable_features:
    :return:
    """
    adata_init = mdata[mod].copy()
    df_chulls_agg.index = df_chulls_agg["well_id"] + "_" + df_chulls_agg["plate_id"]
    # arrange by adata.obs.index
    df_chulls_agg = df_chulls_agg.loc[adata_init.obs.index]
    adata = ad.AnnData(
        X=np.concatenate(
            [
                adata_init.layers["raw"],
                df_chulls_agg.drop(["plate_id", "well_id"], axis=1).to_numpy(),
            ],
            axis=1,
        )
    )
    adata.var_names = list(adata_init.var_names) + list(df_chulls_agg.columns.drop(["plate_id", "well_id"]))
    adata.obs = adata_init.obs.copy()
    adata.layers["raw"] = adata.X.copy()
    sc.pp.scale(adata)
    adata.X[np.isnan(adata.X)] = 0
    if select_variable_features:
        sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, n_comps=32)
    if batch_correction:
        bbknn.bbknn(adata, batch_key="plate_id")
    else:
        sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=1)
    sc.tl.umap(adata, n_components=2, min_dist=0.5)
    mdata.mod[mod] = adata.copy()
    mdata.update()
    return mdata


def merge_org_embeddings(
    mdata_source,
    mdata_target,
    mod="phenocoder_combined",
    batch_correction=True,
    select_variable_features=True,
):
    """
    Merge organoid embeddings
    :param mdata_source:
    :param mdata_target:
    :param mod:
    :param batch_correction:
    :param select_variable_features:
    :return:
    """
    # select modality
    adata_source = mdata_source[mod].copy()
    adata_target = mdata_target[mod].copy()
    # get common labels
    labels_target = adata_target.obs.index.values
    labels_source = adata_source.obs.index.values
    labels = list(set(labels_target).intersection(set(labels_source)))
    idx_target = adata_target.obs.index[adata_target.obs.index.isin(labels)]
    idx_source = adata_source.obs.index[adata_source.obs.index.isin(labels)]
    adata_target = adata_target[idx_target]
    adata_source = adata_source[idx_source]
    # construct merged adata
    df = adata_target.obs.merge(
        adata_source.obs["leiden"],
        left_index=True,
        right_index=True,
        suffixes=("_target", "_source"),
    )
    adata = ad.AnnData(X=np.concatenate([adata_target.layers["raw"], adata_source.layers["raw"]], axis=1))
    adata.obs = df
    var_names_target = [name + "_target" for name in adata_target.var_names]
    var_names_source = [name + "_source" for name in adata_source.var_names]
    adata.var_names = var_names_target + var_names_source
    adata.layers["raw"] = adata.X.copy()
    sc.pp.scale(adata)
    adata.X[np.isnan(adata.X)] = 0
    if select_variable_features:
        sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, n_comps=32)
    if batch_correction:
        bbknn.bbknn(adata, batch_key="plate_id")
    else:
        sc.pp.neighbors(adata, use_rep="X_pca")
    sc.tl.leiden(adata, resolution=1)
    sc.tl.umap(adata, n_components=2, min_dist=0.5)
    return adata


def add_features_to_obs(mdata, source, target, layer=None):
    """
    Add features to obs
    :param mdata:
    :param source:
    :param target:
    :param layer:
    :return:
    """
    if layer is not None:
        df = pd.DataFrame(
            mdata[source].layers[layer],
            columns=mdata[source].var_names,
            index=mdata[source].obs.index,
        )
    else:
        df = pd.DataFrame(
            mdata[source].X,
            columns=mdata[source].var_names,
            index=mdata[source].obs.index,
        )
    mdata.mod[target].obs = pd.concat([mdata[target].obs, df], axis=1)
    mdata.update()
    return mdata


def run_chulls_connected_components(mdata: mu.MuData, clusters: list[str], mod: str = "phenocoder") -> pd.DataFrame:
    """
    Run convex hulls for individual connected components for all samples in dataset for given modality.
    :param mdata:
    :param clusters:
    :param mod:
    :return:
    """
    df_iter = mdata.mod[mod].obs.groupby(["well_id", "plate_id"], observed=True).size().reset_index()
    df_chulls = []
    for well, plate in tqdm(
        zip(df_iter["well_id"], df_iter["plate_id"]),
        total=df_iter.shape[0],
        desc="Computing convex hulls",
    ):
        df_chulls_sample = get_chulls_connected_components(mdata.mod[mod], well, plate, clusters)
        df_chulls.append(df_chulls_sample)
    df_chulls = pd.concat(df_chulls).reset_index(drop=True)
    # group by well and plate -> get mean and sum for all other columns
    agg_dict = {col: ["mean", "sum"] for col in df_chulls.columns if col not in ["well_id", "plate_id"]}
    df_chulls_agg = df_chulls.groupby(["well_id", "plate_id"]).agg(agg_dict)
    # reset row and column multi index
    df_chulls_agg.columns = ["_".join(col).strip() for col in df_chulls_agg.columns.values]
    # add number of connected components -> len df_chulls
    df_chulls_agg["n_chulls"] = df_chulls.shape[0]
    df_chulls_agg.reset_index(inplace=True)
    # left merge with df_iter and fill missing values matches with zeros
    df_chulls_agg = df_iter.loc[:, ["well_id", "plate_id"]].merge(df_chulls_agg, on=["well_id", "plate_id"], how="left")
    df_chulls_agg.fillna(0, inplace=True)
    return df_chulls, df_chulls_agg

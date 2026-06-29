import io as io_builtin
import os

import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib.colors import ListedColormap
from PIL import Image
from scipy.cluster import hierarchy as sch
from skimage import io
from tqdm import tqdm

from phenocoder.plot import generate_examples

plt.rc("pdf", fonttype=42)
sc._settings.settings._vector_friendly = True


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


def read_image(file, shape=(477, 477, 3)):
    if isinstance(file, str) and os.path.exists(file):
        img = io.imread(file)
    else:
        img = np.zeros(shape, dtype=np.uint8)
    return add_black_border(img, 10)


def plot_for_montage(id, adata):
    if isinstance(id, str) and id in adata.obs["id"].values:
        img = plot_organoid(id, adata)
    else:
        # get random id from adata
        id = adata.obs["id"].sample(1, random_state=0).values[0]
        img = np.zeros_like(plot_organoid(id, adata))
    return img


def generate_condition_montages(
    df: pd.DataFrame,
    adata_cycle_01: ad.AnnData,
    adata_cycle_03: ad.AnnData,
    dir_screen: str,
    order_compounds: list = None,
):
    """
    Generate condition montages
    :param df_examples:
    :param adata_cycle_01:
    :param adata_cycle_03:
    :param dir_screen:
    :return:
    """
    # do montage per concentration timepoint combination
    df_example_images = pd.read_csv(f"{dir_screen}/example_images/example_images.csv")
    # select id, file_cycle_1, file_cycle_3 and merge to df_examples
    df_example_images = df_example_images[["id", "file_cycle_1", "file_cycle_3"]]
    df = df.merge(df_example_images, on="id", how="left")
    # generate directory for condition montages
    dir_montages = f"{dir_screen}/montages_conditions"
    os.makedirs(dir_montages, exist_ok=True)
    imgs_timepoints = []
    plots_timepoints = []
    for timepoint in df["timepoint"].unique():
        imgs_01 = []
        imgs_03 = []
        plots_01 = []
        plots_03 = []
        df_tmp = df[df["timepoint"] == timepoint]

        # # count observation per compound
        # df_count = df_tmp['compound'].value_counts().reset_index()
        # # filter out DMSO
        # n_conditions = df_count[df_count['compound'] != 'DMSO']['compound'].unique().shape[0]
        # # reduce number of DMSO to n_conditions
        # df_dmso = df_tmp[df_tmp['compound'] == 'DMSO']
        # # slice df_dmso for n_conditions
        # df_dmso = df_dmso.iloc[:n_conditions]
        # # add to df_tmp
        # df_tmp = pd.concat([df_tmp[df_tmp['compound'] != 'DMSO'], df_dmso])
        # # remove sampled from df
        # df = df[~df['id'].isin(df_dmso['id'])]
        for concentration in tqdm(df_tmp["conc"].unique(), desc=timepoint):
            df_tmp_conc = df_tmp[df_tmp["conc"] == concentration]
            # select unique rows
            df_tmp_conc = df_tmp_conc.drop_duplicates(subset=["id"])
            # check if each compound is represented which is also in order_compounds
            if concentration != "0 µM":
                compounds_condition_present = df_tmp_conc["compound"].unique()
                for compound in order_compounds:
                    if compound not in compounds_condition_present:
                        # add empty row
                        empty_row = pd.DataFrame({"compound": [compound]})
                        df_tmp_conc = pd.concat([df_tmp_conc, empty_row], ignore_index=True)
                # arrange by compounds according to order compounds
                df_tmp_conc = df_tmp_conc.set_index("compound").loc[order_compounds].reset_index()
            assert df_tmp_conc.shape[0] == len(order_compounds)
            files_cycle_01 = df_tmp_conc["file_cycle_1"].values
            files_cycle_03 = df_tmp_conc["file_cycle_3"].values
            ids_tmp = df_tmp_conc["id"].values
            imgs_01.append(np.vstack([read_image(file) for file in files_cycle_01]))
            imgs_03.append(np.vstack([read_image(file) for file in files_cycle_03]))
            plots_01.append(np.vstack([plot_for_montage(id, adata_cycle_01) for id in ids_tmp]))
            plots_03.append(np.vstack([plot_for_montage(id, adata_cycle_03) for id in ids_tmp]))
        imgs_01 = np.hstack(imgs_01)
        imgs_03 = np.hstack(imgs_03)
        plots_01 = np.hstack(plots_01)
        plots_03 = np.hstack(plots_03)
        img = np.vstack([imgs_01, np.zeros((25, imgs_01.shape[1], 3), dtype=np.uint8), imgs_03])
        plot = np.vstack([plots_01, np.zeros((25, plots_01.shape[1], 4), dtype=np.uint8), plots_03])
        # save montage
        io.imsave(f"{dir_montages}/montage_{timepoint}.png", img)
        io.imsave(f"{dir_montages}/montage_{timepoint}_plot.png", plot)
        imgs_timepoints.append(np.hstack([img, np.zeros((img.shape[0], 25, 3), dtype=np.uint8)]))
        plots_timepoints.append(np.hstack([plot, np.zeros((plot.shape[0], 25, 4), dtype=np.uint8)]))
    img = np.hstack(imgs_timepoints)
    plot = np.hstack(plots_timepoints)
    io.imsave(f"{dir_montages}/montage_all.png", img)
    io.imsave(f"{dir_montages}/montage_all_plot.png", plot)


if __name__ == "__main__":
    # load params yaml
    screen = "pilotscreen"
    file = "whole_mount_tumoroid/configs/params.yaml"
    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    # set paths
    dir_screen = "data/processed/pilotscreen"
    dir_adata = f"{dir_screen}/anndata"
    file_org = f"{dir_adata}/mdata_org_combined.h5mu"
    file_cycle_01 = f"{dir_adata}/mdata_cycle-01.h5mu"
    file_cycle_03 = f"{dir_adata}/mdata_cycle-03.h5mu"

    # read mdatas
    mdata_org = mu.read_h5mu(file_org)
    mdata_cycle_01 = mu.read_h5mu(file_cycle_01)
    mdata_cycle_03 = mu.read_h5mu(file_cycle_03)

    # plot umaps
    plot_umap(mdata_cycle_01["phenocoder"], cycle="01", size=0.1)
    plot_umap(mdata_cycle_03["phenocoder"], cycle="03", size=0.1)

    mdata_cycle_01.mod["nuclei"].obs["leiden_phenocoder"] = mdata_cycle_01.mod["phenocoder"].obs["leiden"]
    mdata_cycle_03.mod["nuclei"].obs["leiden_phenocoder"] = mdata_cycle_03.mod["phenocoder"].obs["leiden"]

    plot_dotplot(mdata_cycle_01["nuclei"], cycle="01", remove_cluster=["5"])

    plot_dotplot(mdata_cycle_03["nuclei"], cycle="03", remove_cluster=["5", "6"])

    # plot examples for plate 004, well J08
    io.imsave(
        f"{dir_screen}/example_overlays/004-01/J08/plot.png",
        plot_organoid("J08_004", mdata_cycle_01["phenocoder"], edgecolors="white"),
    )
    io.imsave(
        f"{dir_screen}/example_overlays/004-03/J08/plot.png",
        plot_organoid("J08_004", mdata_cycle_03["phenocoder"], edgecolors="white"),
    )
    # 3d projection plots
    io.imsave(
        f"{dir_screen}/example_overlays/004-01/J08/plot_3d.png",
        plot_organoid(
            "J08_004",
            mdata_cycle_01["phenocoder"],
            project_3d=True,
            edgecolors="grey",
            bg_color="white",
        ),
    )
    io.imsave(
        f"{dir_screen}/example_overlays/004-03/J08/plot_3d.png",
        plot_organoid(
            "J08_004",
            mdata_cycle_03["phenocoder"],
            project_3d=True,
            edgecolors="grey",
            bg_color="white",
        ),
    )

    io.imsave(
        f"{dir_screen}/example_overlays/004-01/J08/plot_3d_cut.png",
        plot_organoid(
            "J08_004",
            mdata_cycle_01["phenocoder"],
            project_3d=True,
            edgecolors="grey",
            bg_color="white",
            cut_open=True,
        ),
    )
    io.imsave(
        f"{dir_screen}/example_overlays/004-03/J08/plot_3d_cut.png",
        plot_organoid(
            "J08_004",
            mdata_cycle_03["phenocoder"],
            project_3d=True,
            edgecolors="grey",
            bg_color="white",
            cut_open=True,
        ),
    )

    # example anndata
    file_examples = "whole_mount_tumoroid/metafiles/positive_ctrls_examples_pilotscreen.csv"
    adata_org_example = mdata_org["phenocoder_combined"].copy()
    df_examples = pd.read_csv(file_examples, dtype={"well_id": str, "plate_id": str})
    df_examples["id"] = df_examples["well_id"] + "_" + df_examples["plate_id"]
    df_examples = df_examples[~df_examples["id"].isin(["E16_004"])]
    # filter adata_org_example by well_id plate_id present in df_examples
    adata_org_example.obs["id"] = (
        adata_org_example.obs["well_id"].astype(str) + "_" + adata_org_example.obs["plate_id"].astype(str)
    )
    adata_org_example = adata_org_example[adata_org_example.obs["id"].isin(df_examples["id"])]

    adata_cycle_01_example = mdata_cycle_01["phenocoder"].copy()
    adata_cycle_01_example.obs["id"] = (
        adata_cycle_01_example.obs["well_id"].astype(str) + "_" + adata_cycle_01_example.obs["plate_id"].astype(str)
    )
    adata_cycle_01_example = adata_cycle_01_example[adata_cycle_01_example.obs["id"].isin(df_examples["id"])]

    adata_cycle_03_example = mdata_cycle_03["phenocoder"].copy()
    adata_cycle_03_example.obs["id"] = (
        adata_cycle_03_example.obs["well_id"].astype(str) + "_" + adata_cycle_03_example.obs["plate_id"].astype(str)
    )
    adata_cycle_03_example = adata_cycle_03_example[adata_cycle_03_example.obs["id"].isin(df_examples["id"])]

    # lut dict
    lut_dict = {
        "01": [(1, 99), (1, 99), (1, 99), (1, 99)],  # DAPI, SDC1, ITGA2, LAMC2
        "03": [(5, 99), (5, 99), (95, 99), (5, 99)],
    }  # DAPI, COL1, KI67, CK19

    # # generate example images for complete organoids
    generate_examples(
        adata_org_example,
        dir_screen=dir_screen,
        lut_dict=lut_dict,
        n=None,
        n_down_sampling=8,
        cycles=("01", "03"),
    )
    # generate montages
    generate_condition_montages(
        df=df_examples,
        dir_screen=dir_screen,
        adata_cycle_01=adata_cycle_01_example,
        adata_cycle_03=adata_cycle_03_example,
        order_compounds=[
            "Bortezomib",
            "Trametinib",
            "SN38",
            "BTT-3033",
            "Gemcitabine",
            "PF-562271",
            "Linsitinib",
            "Paclitaxel",
            "T0070907",
            "VER155008",
            "Erlotinib",
            "LGK-974",
            "Ilomastat",
            "Ac-Gly-BoroPro",
        ],
    )

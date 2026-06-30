import os

import anndata as ad
import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from skimage import io
from tqdm import tqdm

from analysis.utils import add_black_border, plot_dotplot, plot_organoid, plot_umap
from phenocoder.plot import generate_examples

plt.rc("pdf", fonttype=42)
sc._settings.settings._vector_friendly = True


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
    file = "configs/params.yaml"
    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    # set paths
    dir_screen = "data/pilotscreen"
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
    file_examples = "metafiles/positive_ctrls_examples_pilotscreen.csv"
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

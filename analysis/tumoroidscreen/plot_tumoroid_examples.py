from pathlib import Path

import muon as mu
import pandas as pd
import yaml
from analysis.pilotscreen.generate_example_images import (
    plot_dotplot,
    plot_organoid,
    plot_umap,
)
from skimage import io

from phenocoder.plot import generate_examples

if __name__ == "__main__":
    # load params yaml
    screen = "tumoroidscreen"
    file = "whole_mount_tumoroid/configs/params.yaml"
    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    # set paths
    dir_screen = params["dir_screen"]
    dir_adata = f"{dir_screen}/anndata"
    file_org = f"{dir_adata}/mdata_org_combined.h5mu"
    file_cycle_01 = f"{dir_adata}/mdata_cycle-01.h5mu"
    file_cycle_03 = f"{dir_adata}/mdata_cycle-03.h5mu"

    # read mdatas
    mdata_org = mu.read_h5mu(file_org)
    mdata_cycle_01 = mu.read_h5mu(file_cycle_01)
    mdata_cycle_03 = mu.read_h5mu(file_cycle_03)

    # plot umaps
    plot_umap(
        mdata_cycle_01["phenocoder"].uns["adata_sampled"],
        cycle="01",
        dir_screen=dir_screen,
        size=2,
    )

    plot_umap(
        mdata_cycle_03["phenocoder"].uns["adata_sampled"],
        cycle="03",
        dir_screen=dir_screen,
        size=2,
    )

    mdata_cycle_01.mod["nuclei"].obs["leiden_phenocoder"] = mdata_cycle_01.mod["phenocoder"].obs["leiden"]
    mdata_cycle_03.mod["nuclei"].obs["leiden_phenocoder"] = mdata_cycle_03.mod["phenocoder"].obs["leiden"]

    plot_dotplot(mdata_cycle_01["nuclei"], cycle="01", dir_screen=dir_screen)

    plot_dotplot(mdata_cycle_03["nuclei"], cycle="03", dir_screen=dir_screen)

    def generate_spatial_plots_organoid(well, plate, dir_screen, mdata_cycle_01, mdata_cycle_03):
        # plot example organoid for figure
        Path(f"{dir_screen}/example_overlays/{plate}-01/{well}").mkdir(parents=True, exist_ok=True)
        Path(f"{dir_screen}/example_overlays/{plate}-03/{well}").mkdir(parents=True, exist_ok=True)

        io.imsave(
            f"{dir_screen}/example_overlays/{plate}-01/{well}/plot.png",
            plot_organoid(f"{well}_{plate}", mdata_cycle_01["phenocoder"], edgecolors="white"),
        )
        io.imsave(
            f"{dir_screen}/example_overlays/{plate}-03/{well}/plot.png",
            plot_organoid(f"{well}_{plate}", mdata_cycle_03["phenocoder"], edgecolors="white"),
        )
        # 3d projection plots
        io.imsave(
            f"{dir_screen}/example_overlays/{plate}-01/{well}/plot_3d.png",
            plot_organoid(
                f"{well}_{plate}",
                mdata_cycle_01["phenocoder"],
                project_3d=True,
                edgecolors="grey",
                bg_color="white",
            ),
        )
        io.imsave(
            f"{dir_screen}/example_overlays/{plate}-03/{well}/plot_3d.png",
            plot_organoid(
                f"{well}_{plate}",
                mdata_cycle_03["phenocoder"],
                project_3d=True,
                edgecolors="grey",
                bg_color="white",
            ),
        )

        io.imsave(
            f"{dir_screen}/example_overlays/{plate}-01/{well}/plot_3d_cut.png",
            plot_organoid(
                f"{well}_{plate}",
                mdata_cycle_01["phenocoder"],
                project_3d=True,
                edgecolors="grey",
                bg_color="white",
                cut_open=True,
            ),
        )
        io.imsave(
            f"{dir_screen}/example_overlays/{plate}-03/{well}/plot_3d_cut.png",
            plot_organoid(
                f"{well}_{plate}",
                mdata_cycle_03["phenocoder"],
                project_3d=True,
                edgecolors="grey",
                bg_color="white",
                cut_open=True,
            ),
        )

    generate_spatial_plots_organoid("C03", "HM005", dir_screen, mdata_cycle_01, mdata_cycle_03)
    generate_spatial_plots_organoid("H20", "HM004", dir_screen, mdata_cycle_01, mdata_cycle_03)
    generate_spatial_plots_organoid("P23", "HM006", dir_screen, mdata_cycle_01, mdata_cycle_03)
    generate_spatial_plots_organoid("J03", "HM003", dir_screen, mdata_cycle_01, mdata_cycle_03)

    # load examples data
    dir_analysis = "whole_mount_tumoroid/analysis/tumoroidscreen"
    df_examples = pd.read_csv(
        Path(dir_analysis, "tables", "table_cpds_selected_final.csv"),
        dtype={"well_id": str, "plate_id": str},
    )
    df_examples["id"] = df_examples["well_id"] + "_" + df_examples["plate_id"]
    adata_org_example = mdata_org["phenocoder_combined"].copy()

    # filter adata_org_example by well_id plate_id present in df_examples
    adata_org_example.obs["id"] = (
        adata_org_example.obs["well_id"].astype(str) + "_" + adata_org_example.obs["plate_id"].astype(str)
    )
    # sample 8 DMS0 ctrls ids
    dmso_ids = (
        adata_org_example[adata_org_example.obs["negative_control"] == "True"]
        .obs["id"]
        .sample(adata_org_example.obs["id"].isin(df_examples["id"]).sum(), random_state=0)
        .tolist()
    )
    adata_org_example = adata_org_example[adata_org_example.obs["id"].isin(dmso_ids + df_examples["id"].tolist())]

    # lut dict
    lut_dict = {
        "01": [(1, 99), (1, 99), (1, 99), (1, 99)],  # DAPI, SDC1, ITGA2, LAMC2
        "03": [(5, 99), (5, 99), (95, 99), (5, 99)],
    }  # DAPI, COL1, KI67, CK19

    # generate example images for complete organoids
    generate_examples(
        adata_org_example,
        dir_screen=dir_screen,
        lut_dict=lut_dict,
        n=None,
        n_down_sampling=8,
        scale_bar=310,
        cycles=("01", "03"),
    )

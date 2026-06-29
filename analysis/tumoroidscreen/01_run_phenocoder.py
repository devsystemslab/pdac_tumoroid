from pathlib import Path

import pandas as pd
import yaml

from phenocoder.cluster import run_clustering_pipeline
from phenocoder.embedding import run_organoid_embedding
from phenocoder.features import run_feature_processing
from phenocoder.spatial import run_spatial_feature_processing

if __name__ == "__main__":
    screen = "tumoroidscreen"
    file = "whole_mount_tumoroid/configs/params.yaml"
    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    df_plate_layouts = pd.read_csv(
        Path(params["dir_screen"], "plate_layout.csv"),
        dtype={"plate": str, "well": str},
    )
    df_plate_layouts.rename(
        columns={"plate": "plate_id", "well": "well_id"}, inplace=True
    )
    df_plate_layouts = df_plate_layouts.map(str)
    dir_results = Path(params["dir_screen"], "anndata_include_all")
    dir_results.mkdir(parents=True, exist_ok=True)

    print("Running phenocoder pipeline for registered data...")
    mdata = run_feature_processing(
        plates=params["plates"],
        markers=params["markers"],
        dir_screen=params["dir_screen"],
        input_type=params["input_type"],
        registered=params["registered"],
        dir_models=params["phenocoder"]["dir_models"],
        model_dict=params["phenocoder"]["models"],
        qc_score_threshold=params["qc"]["qc_score_threshold"],
        qc_distance_threshold=params["qc"]["qc_distance_threshold"],
    )

    mdata = run_clustering_pipeline(
        mdata=mdata,
        use_gpu=False,
        n_comps_pca=params["phenocoder"]["n_comps_pca"],
        res=params["phenocoder"]["cluster_res"],
    )
    mdata.write_h5mu(Path(dir_results, "mdata_registered.h5mu"))

    spatial_dict = run_spatial_feature_processing(mdata)

    mdata_org = run_organoid_embedding(
        spatial_dict=spatial_dict,
        df_plate_layouts=df_plate_layouts,
        n_comps_pca=params["phenocoder"]["n_comps_pca"],
        res=params["phenocoder"]["cluster_res"],
        batch_correction=False,
        confounder=None,
    )

    mdata_org.write_h5mu(Path(dir_results, "mdata_org_default_registered.h5mu"))

    # run phenocoder pipeline for each cycle individually
    cycles = {"01": "source", "03": "target"}
    for cycle in cycles.keys():
        print(f"Running phenocoder pipeline for cycle {cycle}...")
        markers = {
            k.replace(f"{cycles[cycle]}_", ""): v
            for k, v in params["markers"].items()
            if cycles[cycle] in k
        }
        mdata = run_feature_processing(
            plates=params["plates"],
            markers=markers,
            dir_screen=params["dir_screen"],
            input_type=params["input_type"],
            registered=False,
            dir_models=params["phenocoder"]["dir_models"],
            model_dict=params["phenocoder"]["models"],
            cycle=cycle,
        )

        mdata = run_clustering_pipeline(
            mdata=mdata,
            use_gpu=False,
            n_comps_pca=params["phenocoder"]["n_comps_pca"],
            res=params["phenocoder"]["cluster_res"],
        )

        mdata.write_h5mu(Path(dir_results, f"mdata_cycle-{cycle}.h5mu"))

        spatial_dict = run_spatial_feature_processing(mdata)

        mdata_org = run_organoid_embedding(
            spatial_dict=spatial_dict,
            df_plate_layouts=df_plate_layouts,
            n_comps_pca=params["phenocoder"]["n_comps_pca"],
            res=params["phenocoder"]["cluster_res"],
            batch_correction=False,
            confounder=None,
        )

        mdata_org.write_h5mu(Path(dir_results, f"mdata_org_default_cycle-{cycle}.h5mu"))

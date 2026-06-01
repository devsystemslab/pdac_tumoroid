import pickle
from pathlib import Path

import anndata as ad
import muon as mu
import pandas as pd
import yaml

from whole_mount_tumoroid.phenocoder.cluster import run_clustering_pipeline
from whole_mount_tumoroid.phenocoder.embedding import run_organoid_embedding
from whole_mount_tumoroid.phenocoder.features import run_feature_processing
from whole_mount_tumoroid.phenocoder.spatial import run_spatial_feature_processing

from whole_mount_tumoroid.analysis.timecourse.run_mlp_training import train_mlp
from whole_mount_tumoroid.analysis.timecourse.run_mlp_imputation import mlp_impute

if __name__ == "__main__":
    screen = "timecourse"
    file = "whole_mount_tumoroid/configs/params.yaml"

    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    print(params)

    stain_metadata = pd.read_csv('/pstore/data/ihb-g-deco/USERS/schulzp9/tumoroid/metafiles/timecourse_stainings_metadata.csv')
    stain_metadata['staining_set'] = stain_metadata['staining_set'].astype(str)
    stains = stain_metadata.stain.unique()
    stain_dict = {
        set_id: dict(group[['channel', 'stain']].values) 
        for set_id, group in stain_metadata.groupby('staining_set')
    }

    df_plate_layout = pd.read_csv(Path(params["dir_screen"], "timecourse_layout.csv"))
    df_plate_layout = df_plate_layout.melt(
        id_vars=["row"], var_name="col", value_name="staining_set"
    )
    df_plate_layout["column"] = df_plate_layout["col"].str.zfill(2)
    df_plate_layout["row_num"] = df_plate_layout["row"].apply(
        lambda x: ord(x) - ord("A") + 1
    )
    df_plate_layout["well_id"] = df_plate_layout["row"] + df_plate_layout["column"]
    df_plate_layout = df_plate_layout.sort_values(by=["well_id"])

    dir_screen = params["dir_screen"]
    plates = params["plates"]
    df_plates = pd.concat(
        [
            pd.read_csv(
                f"{dir_screen}/{plate}/plate_information.csv", dtype={plate: str}
            ).assign(plate=plate)
            for plate in plates
        ]
    )

    df_plate_layouts = df_plate_layout.merge(df_plates, how="cross")
    df_plate_layouts["plate_id"] = df_plate_layouts["plate"]
    df_plate_layouts = df_plate_layouts.map(str)

    dir_results = Path(params["dir_screen"], "anndata")
    # dir_results.mkdir(parents=True, exist_ok=True)
    print("Running phenocoder pipeline ")
    mdata = run_feature_processing(
        plates=params["plates"],
        markers=params["markers"],
        dir_screen=params["dir_screen"],
        input_type=params["input_type"],
        registered=params["registered"],
        dir_models=params["phenocoder"]["dir_models"],
        model_dict=params["phenocoder"]["models"],
        cycle=params["phenocoder"]["models"]["source"]["cycle"],
        # qc_score_threshold=params['qc']['qc_score_threshold'],
        # qc_distance_threshold=params['qc']['qc_distance_threshold'],
        channels=["01"],
    )

    mdata.write_h5mu(Path(dir_results, "mdata_temp.h5mu"))

    mdata = mu.read_h5mu(Path(dir_results, "mdata_temp.h5mu"))

    mdata = run_clustering_pipeline(
        mdata=mdata,
        use_gpu=False,
        n_comps_pca=params["phenocoder"]["n_comps_pca"],
        res=params["phenocoder"]["cluster_res"],
    )

    mdata.write_h5mu(Path(dir_results, "mdata_registered.h5mu"))

    #mdata = mu.read_h5mu(Path(dir_results, 'mdata_registered.h5mu'))

    #train mlp for imputation
    train_mlp(params=params, 
                mdata=mdata, 
                mlp_dir=params['mlp']['dir_models'], 
                base_modality=params['mlp']['base_modality'], 
                imputations=params['mlp']['imputations'], 
                epochs=2,
                stains=stains,
                stain_dict=stain_dict,
                df_plate_layouts=df_plate_layouts)

    mdata_imputed = mlp_impute(
                        mdata=mdata, 
                        mlp_dir=params['mlp']['dir_models'], 
                        base_modality=params['mlp']['base_modality'], 
                        imputations=params['mlp']['imputations'],
                        models=params['mlp']['models'],
                        n_comps_pca=params['phenocoder']['n_comps_pca']['imputed'],
                        resolution=params['phenocoder']['cluster_res']['imputed'],
                        stains=stains
                    )

    mdata_imputed.write_h5mu(Path(dir_results, 'mdata_imputed.h5mu'))
    
    #mdata_imputed = mu.read_h5mu(Path(dir_results, 'mdata_imputed.h5mu'))

    spatial_dict = run_spatial_feature_processing(mdata_imputed)

    print("Run organoid embedding")
    mdata_org = run_organoid_embedding(
        spatial_dict=spatial_dict,
        df_plate_layouts=df_plate_layouts,
        n_comps_pca=params["phenocoder"]["n_comps_pca"],
        res=params["phenocoder"]["cluster_res"],
        batch_correction=False,
        confounder=None,
        combine_modalities=True,
    )

    mdata_org.write_h5mu(Path(dir_results, f"mdata_org.h5mu"))
    print("Job completed")

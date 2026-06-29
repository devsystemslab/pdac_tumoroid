from pathlib import Path

import pandas as pd
import yaml

from phenocoder.cluster import run_clustering_pipeline
from phenocoder.embedding import run_organoid_embedding
from phenocoder.features import run_feature_processing
from phenocoder.spatial import run_spatial_feature_processing


def generate_plate_layout(dir_screen: str, dir_metafiles: str):
    """
    Generate plate layout
    :param dir_screen:
    :param dir_metafiles:
    :return:
    """
    # add meta information
    df_caf_lookup = pd.read_csv(Path(dir_metafiles, "egfr_tidy_patient_caf_lookup.csv"), dtype={"plate": str})
    df_patient_meta = pd.read_csv(Path(dir_metafiles, "egfr_tidy_patient_meta.csv")).rename(
        columns={"patient": "cancer"}
    )
    # all to string
    df_patient_meta.loc[:, df_patient_meta.dtypes == "int64"] = df_patient_meta.loc[
        :, df_patient_meta.dtypes == "int64"
    ].astype(str)
    df_drugs = pd.read_csv(Path(dir_metafiles, "egfr_tidy_platelayout_drugs_wide.csv"))
    # pivot long index = row
    df_drugs = df_drugs.melt(id_vars="row", var_name="col", value_name="drug")
    # add convert col into double digits style
    df_drugs = df_drugs.assign(col=df_drugs["col"].apply(lambda x: f"{int(x):02d}")).assign(
        well=lambda x: x["row"] + x["col"]
    )

    df_patient_caf = pd.read_csv(Path(dir_metafiles, "egfr_tidy_platelayout_patient_caf_wide.csv"))
    df_patient_caf = df_patient_caf.melt(id_vars="row", var_name="col", value_name="cancer_x_caf")
    df_patient_caf = df_patient_caf.assign(col=df_patient_caf["col"].apply(lambda x: f"{int(x):02d}")).assign(
        well=lambda x: x["row"] + x["col"]
    )
    # merge with df_caf_lookup on cancer_x_caf
    df_patient_caf = pd.merge(df_patient_caf, df_caf_lookup, on="cancer_x_caf", how="left")
    df_patient_caf["cancer_x_caf"] = df_patient_caf["cancer_x_caf"].astype(str)
    # merge df_patient_meta
    df_patient_caf = pd.merge(df_patient_caf, df_patient_meta, on="cancer", how="left")
    df_plate_layout = pd.merge(df_patient_caf, df_drugs, on=["well", "row", "col"], how="left")
    df_plate_layout.to_csv(Path(dir_screen, "plate_layout.csv"), index=False)


if __name__ == "__main__":
    screen = "egfr"
    file = "/pstore/home/harmelc/tumoroid/whole_mount_tumoroid/configs/params.yaml"
    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    generate_plate_layout(
        dir_screen=params["dir_screen"],
        dir_metafiles="/pstore/home/harmelc/tumoroid/whole_mount_tumoroid/metafiles",
    )
    df_plate_layouts = pd.read_csv(
        Path(params["dir_screen"], "plate_layout.csv"),
        dtype={"plate": str, "well": str},
    )
    df_plate_layouts.rename(columns={"plate": "plate_id", "well": "well_id"}, inplace=True)
    df_plate_layouts = df_plate_layouts.map(str)

    mdata = run_feature_processing(
        plates=params["plates"],
        markers=params["markers"],
        dir_screen=params["dir_screen"],
        input_type=params["input_type"],
        registered=params["registered"],
        dir_models=params["phenocoder"]["dir_models"],
        model_dict=params["phenocoder"]["models"],
        cycle="01",
    )

    mdata = run_clustering_pipeline(
        mdata=mdata,
        use_gpu=False,
        n_comps_pca=params["phenocoder"]["n_comps_pca"],
        res=params["phenocoder"]["cluster_res"],
    )

    mdata.write_h5mu(Path(params["dir_screen"], "anndata", "mdata.h5mu"))

    spatial_dict = run_spatial_feature_processing(mdata)

    mdata_org = run_organoid_embedding(
        spatial_dict=spatial_dict,
        df_plate_layouts=df_plate_layouts,
        n_comps_pca=params["phenocoder"]["n_comps_pca"],
        res=params["phenocoder"]["cluster_res"],
        batch_correction=False,
        confounder=None,
    )

    mdata_org.write_h5mu(Path(params["dir_screen"], "anndata", "mdata_org.h5mu"))

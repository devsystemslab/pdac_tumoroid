from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd

# load mdata
dir_adata = "data/processed/timecourse/anndata"
mdata = mu.read_h5mu(Path(dir_adata, "mdata_org.h5mu"))


df = (
    mdata.mod["all_combined"]
    .obs.join(mdata.mod["all_combined"].to_df(), how="left")
    .copy()
)
# make output directory
Path(dir_adata, "exported_csv").mkdir(parents=True, exist_ok=True)
df.to_csv(Path(dir_adata, "exported_csv", "timecourse_features.csv"))

df = (
    mdata.mod["all_combined"]
    .obs.join(mdata.mod["all_combined"].to_df(layer="raw"), how="left")
    .copy()
)
# make output directory
Path(dir_adata, "exported_csv").mkdir(parents=True, exist_ok=True)
df.to_csv(Path(dir_adata, "exported_csv", "timecourse_features_raw.csv"))

# feature importance
adata = mdata["all_combined"].copy()
adata.uns["dot_product_pc_X"] = np.dot(adata.obsm["X_pca"].T, adata.X)
# z-score
adata.uns["z_score"] = (
    adata.uns["dot_product_pc_X"] - adata.uns["dot_product_pc_X"].mean(axis=0)
) / (adata.uns["dot_product_pc_X"].std(axis=0) + 0.00000000000001)
# weight pcs by explained variance and aggregate absolute dot product values
adata.uns["dot_product_pc_X_weighted"] = np.dot(
    np.abs(adata.uns["z_score"]).T, adata.uns["pca"]["variance_ratio"]
)
# generate a dataframe with names from X and dot_product_pc_X_weighted as column
df = pd.DataFrame(
    {"feature": adata.var_names, "score": adata.uns["dot_product_pc_X_weighted"]}
)
# rank features by score
df["rank"] = df["score"].rank(ascending=True)
# for features with same rank -> sample one and discard the others
df = df.sort_values(by="rank").drop_duplicates(subset="rank", keep="first")
# re-rank
df["rank"] = df["score"].rank(ascending=True)
# write df as csv
df.to_csv(
    Path(dir_adata, "exported_csv", "timecourse_feature_importance.csv"), index=False
)

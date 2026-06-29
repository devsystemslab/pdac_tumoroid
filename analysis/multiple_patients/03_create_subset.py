import bbknn
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

screen = "egfr"
file = "/pstore/home/harmelc/tumoroid_screen/whole_mount_tumoroid/configs/params.yaml"
with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

# read in mutation data
df = pd.read_csv(
    "/pstore/home/harmelc/tumoroid_screen/whole_mount_tumoroid/metafiles/organoid_mutation.tsv",
    sep="\t",
)
# transpose
df = df.set_index("mut").T
# rename index to patient_id and reset index
df.index.name = "cancer"
df = df.reset_index()
# drop mut
df.columns.name = None


dir_adata = f"{params['dir_screen']}/anndata"

mdata_org = mu.read_h5mu(f"{dir_adata}/mdata_org.h5mu")

lines = ["P382", "P388", "P506", "P585"]
for mod in mdata_org.mod.keys():
    adata = mdata_org.mod[mod].copy()

    # filter where cancer == caf
    # adata = adata[adata.obs["cancer"].astype(str) == adata.obs["caf"].astype(str)]
    adata = adata[adata.obs["cancer"].astype(str).isin(lines)]
    adata = adata[adata.obs["caf"].astype(str).isin(lines)]
    # filter out plate_id == '007'
    adata = adata[adata.obs["plate_id"] != "007"]

    adata.X = adata.layers["raw"].copy()
    sc.pp.scale(adata)
    adata.X[np.isnan(adata.X)] = 0
    sc.pp.highly_variable_genes(adata)
    sc.tl.pca(adata, n_comps=32, use_highly_variable=True)
    bbknn.bbknn(adata, batch_key="plate_id")
    sc.tl.leiden(adata, resolution=0.5)
    sc.tl.umap(adata, n_components=2, min_dist=0.1)
    # adata.obs = pd.merge(adata.obs, df, how='left', on='cancer')
    mdata_org.mod[mod] = adata

# Update MuData to reflect changes in modalities
mdata_org.update()
# write to new file
mdata_org.write_h5mu(f"{dir_adata}/mdata_subset.h5mu")

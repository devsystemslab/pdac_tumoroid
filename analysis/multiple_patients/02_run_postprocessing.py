import bbknn
import muon as mu
import pandas as pd
import scanpy as sc
import yaml

from analysis.utils import run_umap_gpu

screen = "egfr"
file = "/pstore/home/harmelc/tumoroid/whole_mount_tumoroid/configs/params.yaml"
with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

# read in mutation data
df = pd.read_csv("/pstore/home/harmelc/tumoroid/whole_mount_tumoroid/metafiles/organoid_mutation.tsv", sep="\t")
# transpose
df = df.set_index("mut").T
# rename index to patient_id and reset index
df.index.name = "cancer"
df = df.reset_index()
# drop mut
df.columns.name = None

dir_adata = f"{params['dir_screen']}/anndata"
mdata = mu.read_h5mu(f"{dir_adata}/mdata.h5mu")
mdata = run_umap_gpu(mdata, rep="X_pca")
mdata.write_h5mu(f"{dir_adata}/mdata.h5mu")

mdata_org = mu.read_h5mu(f"{dir_adata}/mdata_org.h5mu")
for mod in mdata_org.mod_names:
    bbknn.bbknn(mdata_org[mod], batch_key="plate_id")
    sc.tl.leiden(mdata_org[mod], resolution=0.5)
    sc.tl.umap(mdata_org[mod], min_dist=0.5, n_components=2)
    mdata_org.mod[mod].obs = pd.merge(mdata_org.mod[mod].obs, df, how="left", on="cancer")
mdata_org.write_h5mu(f"{dir_adata}/mdata_org.h5mu")

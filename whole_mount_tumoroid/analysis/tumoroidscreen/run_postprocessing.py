import bbknn
import muon as mu
import scanpy as sc
import yaml

from whole_mount_tumoroid.analysis.utils import (
    add_chull_stats_to_mdata_org,
    merge_org_embeddings,
    run_chulls_connected_components,
    run_umap_gpu,
)

screen = "tumoroidscreen"
file = "whole_mount_tumoroid/configs/params.yaml"
with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

# set paths
dir_adata = f"{params['dir_screen']}/anndata"
file_registered = f"{dir_adata}/mdata_registered.h5mu"
file_cycle_01 = f"{dir_adata}/mdata_cycle-01.h5mu"
file_cycle_03 = f"{dir_adata}/mdata_cycle-03.h5mu"

# read mdatas
mdata_reg = mu.read_h5mu(file_registered)
mdata_cycle_01 = mu.read_h5mu(file_cycle_01)
mdata_cycle_03 = mu.read_h5mu(file_cycle_03)

# rerun umap embedding for complete datasets
mdata_reg = run_umap_gpu(mdata_reg)
mdata_1 = run_umap_gpu(mdata_cycle_01)
mdata_2 = run_umap_gpu(mdata_cycle_03)

# write mdatas
mdata_reg.write_h5mu(file_registered)
mdata_1.write_h5mu(file_cycle_01)
mdata_2.write_h5mu(file_cycle_03)

# organoid embeddings
# set paths
file_org_cycle_01 = f"{dir_adata}/mdata_org_default_cycle-01.h5mu"
file_org_cycle_03 = f"{dir_adata}/mdata_org_default_cycle-03.h5mu"
file_org_reg = f"{dir_adata}/mdata_org_default_registered.h5mu"

# read mdatas
mdata_org_cycle_01 = mu.read_h5mu(file_org_cycle_01)
mdata_org_cycle_03 = mu.read_h5mu(file_org_cycle_03)
mdata_org_reg = mu.read_h5mu(file_org_reg)

# process registered organoid embeddings
for mod in mdata_org_reg.mod_names:
    sc.external.pp.harmony_integrate(
        mdata_org_reg[mod], key="plate_id", max_iter_harmony=30
    )
    bbknn.bbknn(mdata_org_reg[mod], batch_key="plate_id", use_rep="X_pca_harmony")
    sc.tl.leiden(mdata_org_reg[mod], resolution=1)
    sc.tl.umap(mdata_org_reg[mod], min_dist=0.1, n_components=2)
mdata_org_reg.update()
mdata_org_reg.write_h5mu(file_org_reg)

# segment ducts
# TODO: checkout which clusters are for ducts!
df_chulls_cycle_01, df_chulls_agg_cycle_01 = run_chulls_connected_components(
    mdata_cycle_01, clusters=["2", "3"]
)
df_chulls_cycle_03, df_chulls_agg_cycle_03 = run_chulls_connected_components(
    mdata_cycle_03, clusters=["4", "3"]
)

# update organoid embeddings
for mod in ["phenocoder", "phenocoder_combined"]:
    mdata_org_cycle_01 = add_chull_stats_to_mdata_org(
        mdata_org_cycle_01, df_chulls_agg_cycle_01, mod=mod
    )
    mdata_org_cycle_03 = add_chull_stats_to_mdata_org(
        mdata_org_cycle_03, df_chulls_agg_cycle_03, mod=mod
    )

# write mdatas
mdata_org_cycle_01.write_h5mu(f"{dir_adata}/mdata_org_chull_cycle-01.h5mu")
mdata_org_cycle_03.write_h5mu(f"{dir_adata}/mdata_org_chull_cycle-03.h5mu")

# merge organoid embeddings and reruns phenotypic embedding
adata_dict = {}
for mod_key in mdata_org_cycle_01.mod_names:
    mdata_org_combined = merge_org_embeddings(
        mdata_org_cycle_01, mdata_org_cycle_03, mod=mod_key
    )
    adata_dict[mod_key] = mdata_org_combined.copy()
mdata_org_combined = mu.MuData(adata_dict)

# write mdata
file_combined = f"{dir_adata}/mdata_org_combined.h5mu"
mdata_org_combined.write_h5mu(file_combined)

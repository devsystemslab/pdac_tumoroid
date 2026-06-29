import muon as mu

from analysis.utils import (
    add_chull_stats_to_mdata_org,
    merge_org_embeddings,
    run_chulls_connected_components,
    run_umap_gpu,
)

# set paths
dir_adata = "data/processed/inhibitors/anndata"
file_registered = f"{dir_adata}/mdata_registered.h5mu"
file_cycle_01 = f"{dir_adata}/mdata_cycle-01.h5mu"
file_cycle_03 = f"{dir_adata}/mdata_cycle-03.h5mu"

# read mdatas
mdata_reg = mu.read_h5mu(file_registered)
mdata_cycle_01 = mu.read_h5mu(file_cycle_01)
mdata_cycle_03 = mu.read_h5mu(file_cycle_03)

# rerun umap embedding for complete datasets
mdata_reg = run_umap_gpu(mdata_reg)
mdata_cycle_01 = run_umap_gpu(mdata_cycle_01)
mdata_cycle_03 = run_umap_gpu(mdata_cycle_03)

# write mdatas
mdata_reg.write_h5mu(file_registered)
mdata_cycle_01.write_h5mu(file_cycle_01)
mdata_cycle_03.write_h5mu(file_cycle_03)

# organoid embeddings
file_org_cycle_01 = f"{dir_adata}/mdata_org_default_cycle-01.h5mu"
file_org_cycle_03 = f"{dir_adata}/mdata_org_default_cycle-03.h5mu"

# read mdatas
mdata_org_cycle_01 = mu.read_h5mu(file_org_cycle_01)
mdata_org_cycle_03 = mu.read_h5mu(file_org_cycle_03)

# segment ducts
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

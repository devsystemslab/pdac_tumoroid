import muon as mu

from whole_mount_tumoroid.analysis.utils import (
    add_chull_stats_to_mdata_org,
    run_chulls_connected_components,
    run_umap_gpu,
)

# set paths
dir_adata = "data/processed/timecourse/anndata"

file_registered = f"{dir_adata}/mdata_registered.h5mu"

# read mdatas
mdata_reg = mu.read_h5mu(file_registered)

# organoid embeddings
file_org = f"{dir_adata}/mdata_org.h5mu"

# read mdatas
mdata_org = mu.read_h5mu(file_org)

# segment ducts
df_chulls_org, df_chulls_agg_org = run_chulls_connected_components(
    mdata_reg, clusters=["0", "1"]
)

# update organoid embeddings
for mod in ["phenocoder", "phenocoder_combined"]:
    mdata_org = add_chull_stats_to_mdata_org(mdata_org, df_chulls_agg_org, mod=mod)

# write mdatas
mdata_org.write_h5mu(f"{dir_adata}/mdata_org_chull.h5mu")

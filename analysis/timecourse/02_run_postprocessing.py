import muon as mu
import pandas as pd

from analysis.utils import (
    add_chull_stats_to_mdata_org,
    run_chulls_connected_components,
)

# set paths
dir_adata = "/pstore/data/ihb-tumoroid/data/processed/timecourse/anndata"

file_registered = "/pmount/projects/site/pred/ihb-tumoroid/data/processed/timecourse/anndata/mdata_registered_imputed_mlp_normalized.h5mu"

# read mdatas
mdata_reg = mu.read_h5mu(file_registered)

# organoid embeddings
file_org = "/pstore/data/ihb-tumoroid/data/processed/timecourse/anndata/mdata_org_mlp_normalized.h5mu"

# read mdatas
mdata_org = mu.read_h5mu(file_org)


chulls_mod_cluster = []
# update organoid embeddings
for mod in ["phenocoder_msg_nuclei_imputed", "phenocoder_msg_neighbors_imputed"]:
    for cluster in mdata_reg.mod[mod].obs["leiden"].unique():
        print(mod, cluster)
        # segment ducts
        df_chulls_org, df_chulls_agg_org = run_chulls_connected_components(
            mdata_reg, clusters=[cluster], mod=mod
        )
        chulls_mod_cluster.append(df_chulls_agg_org)

df_all = pd.concat(
    [df.set_index(["well_id", "plate_id"]) for df in chulls_mod_cluster],
    axis=1,
).reset_index(drop=False)


mdata_org = add_chull_stats_to_mdata_org(
    mdata_org, df_all, mod="msg_imputed_combined", batch_correction=False
)
mdata_org = mu.MuData(mdata_org.mod)
# write mdatas
mdata_org.write_h5mu(f"{dir_adata}/mdata_org_chull_mlp_normalized.h5mu")

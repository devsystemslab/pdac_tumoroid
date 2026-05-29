import yaml
import muon as mu
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
from sklearn.metrics.cluster import pair_confusion_matrix

import warnings
warnings.filterwarnings("ignore")

from whole_mount_tumoroid.phenocoder.cluster import clustering, project_clustering, build_search_tree, majority_voting, get_accuracy

from pathlib import Path

plt.rc("pdf", fonttype=42)

screen = 'inhibitors'
file = "/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/configs/params.yaml"

with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

dir_screen = params["dir_screen"]
dir_results = Path(params["dir_screen"], "anndata")

cycle = ['01', '03']
mods = ['phenocoder', 'phenocoder_msg']

############# PROJECT LABELS ################################################

frac = 1.0
collected = {}

for cyc in cycle:
    print(cyc)
    mdata_orig_cyc = mu.read_h5mu(
        Path(dir_results, f"mdata_cycle-{cyc}.h5mu")
    )
    mdata_new_cyc = mu.read_h5mu(
        Path(dir_results, f"mdata_cycle-{cyc}_inhibitors_noplatecondition.h5mu")
    )

    subset_orig = mu.MuData({m: mdata_orig_cyc.mod[m] for m in mods})
    subset_new = mu.MuData({m: mdata_new_cyc.mod[m] for m in mods})

    keep_cyc = subset_new.mod["phenocoder"].obs.query(f"plate_id == '004'").index

    for mod in mods:
        print(mod)
        adata = mdata_new_cyc.mod[mod].copy()

        adata = clustering(adata, dry_run=True, run_pca=True, n_comps=params['phenocoder']['n_comps_pca'][mod], use_gpu=False)
        plate4_subset = adata[keep_cyc]

        adata_sampled = plate4_subset[np.random.choice(plate4_subset.obs.index, int(frac * plate4_subset.shape[0]), replace=False)].copy()
        adata = project_clustering(adata, adata_sampled, ref_cluster_key='leiden', cluster_key_added='plate_4_projected')
        accuracy = get_accuracy(adata, adata_sampled, ref_cluster_key='leiden', cluster_key_added='plate_4_projected')
        adata.uns['label_transfer_accuracy_plate_4_projected'] = accuracy
        adata.obs['plate_4_projected'] = adata.obs['plate_4_projected'].astype(str)
        adata.obs['plate_4_projected'] = pd.Categorical(adata.obs['plate_4_projected'])
        adata.uns['adata_sampled_plate_4_projected'] = adata_sampled

        collected[f"cycle{cyc}_{mod}"] = adata

############# SAVE ###########################################################

mdata_combined = mu.MuData(collected)
mdata_combined.write(Path(dir_results, f"mdata_plate4_projected.h5mu"))

mdata_combined = mu.read_h5mu(Path(dir_results, "mdata_plate4_projected.h5mu"))

############# PLOT CONFUSION MATRIX ##########################################

plates = ['004', '005']
subsets = {}
titles = {}
def get_subset(adata, plate_id):
    return adata[adata.obs["plate_id"] == plate_id]

for cyc in cycle:
    mdata_orig_cyc = mu.read_h5mu(
        Path(dir_results, f"mdata_cycle-{cyc}.h5mu")
    )

    for mod in mods:
        adata = mdata_combined.mod[f"cycle{cyc}_{mod}"]
        adata_orig = mdata_orig_cyc.mod[mod].copy()

        subsets = {p: (get_subset(adata_orig, p), get_subset(adata, p)) for p in plates}
        subsets["all"] = (adata_orig, adata)

        titles = {p: f"Plate {p}" for p in plates}
        titles["all"] = "All observations"

        fig, axes = plt.subplots(1, len(subsets), figsize=(24, 8))

        for col, (key, (orig, new)) in enumerate(subsets.items()):
            confusion = pd.crosstab(
                orig.obs["leiden"],
                new.obs["plate_4_projected"],
                rownames=["leiden"],
                colnames=["plate_4_projected"]
            )
            confusion_norm = confusion.div(confusion.sum(axis=1), axis=0)

            sns.heatmap(
                confusion_norm,
                ax=axes[col],
                cmap="Blues",
                annot=confusion,
                fmt="d",
                linewidths=0.3,
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Proportion", "shrink": 0.8},
                annot_kws={"size": 7},
            )
            axes[col].set_title(titles[key])
            axes[col].set_xticklabels(axes[col].get_xticklabels(), rotation=0, ha="right")
            axes[col].set_yticklabels(axes[col].get_yticklabels(), rotation=0)

        plt.suptitle(f"leiden vs plate_4_projected (row-normalized) | cycle {cyc} - {mod}", y=1.02)
        plt.tight_layout()
        #plt.savefig(f"../timecourse/plots_new/confusion_leiden_vs_projected_cycle{cyc}_{mod}.pdf", bbox_inches="tight")
        plt.show()
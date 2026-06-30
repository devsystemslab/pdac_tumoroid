from pathlib import Path

import muon as mu
import numpy as np
import squidpy as sq
from scipy.sparse import csr_array
from tqdm import tqdm


def message_passing(adata, radii):
    # calculate knn graph in physical space
    adata.obsm["spatial"] = adata.obs[["z", "centroid-0", "centroid-1"]].values.copy()
    for radius in tqdm(radii, desc="Calculating spatial neighbors"):
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", spatial_key="spatial")
        A = adata.obsp["spatial_connectivities"].copy()
        A = A + csr_array(np.diag(np.ones(A.shape[0])))
        # weight A with inverse degree matrix
        D = np.array(A.sum(axis=1)).flatten()
        D_inv = np.power(D, -1)
        D_inv[np.isinf(D_inv)] = 0
        D_inv = np.diag(D_inv)
        A = A.dot(D_inv)
        adata.layers[f"message_passing_radius_{radius}"] = np.dot(A, adata.X)
    return adata


def main(
    file="mdata_cycle-01.h5mu",
    dir_adata="data/pilotscreen/anndata",
):
    mdata = mu.read_h5mu(f"{dir_adata}/{file}")
    adata = mdata.mod["nuclei"]
    # get unique sample combinations: well_id plate_id
    samples = adata.obs[["well_id", "plate_id"]].drop_duplicates().reset_index(drop=True)
    dir_results = Path(dir_adata, "sensitivity_analysis_results")
    dir_results.mkdir(exist_ok=True)
    for _, row in samples.iterrows():
        well_id = row["well_id"]
        plate_id = row["plate_id"]
        adata_sub = adata.copy()
        adata_sub = adata[adata.obs["well_id"] == well_id]
        adata_sub = adata_sub[adata_sub.obs["plate_id"] == plate_id]
        adata_sub = message_passing(adata_sub, radii=[5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 200])
        adata_sub.write_h5ad(Path(dir_results, f"{well_id}_{plate_id}_sensitivity_analysis.h5ad"))


if __name__ == "__main__":
    main()

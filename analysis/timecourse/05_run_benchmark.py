import argparse
from pathlib import Path

import muon as mu

from benchmarking import benchmarking

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--mode", default="client")
    parser.add_argument("--port", default=40467)
    parser.add_argument("--host", default="client")
    args = parser.parse_args()
    # set paths
    dir_adata = "/pstore/data/ihb-tumoroid/data/processed/timecourse/anndata"

    dir_results = Path(dir_adata, "benchmarking")
    dir_results.mkdir(parents=True, exist_ok=True)

    file = f"{dir_adata}/mdata_registered_imputed_mlp_clean.h5mu"
    # read mdatas
    mdata = mu.read_h5mu(file)

    mod_keys = {
        "phenocoder": ("X", "leiden_phenocoder"),
        "phenocoder_msg": ("X", "leiden_phenocoder_msg"),
        "nuclei": ("pca", "leiden_nuclei"),
        "nuclei_msg": ("pca", "leiden_nuclei_msg"),
        "phenocoder_msg_nuclei_imputed": ("X", "leiden_imputed_nuclei"),
        "phenocoder_msg_neighbors_imputed": ("X", "leiden_imputed_neighbors"),
    }

    for mod_key in mod_keys:
        mdata.mod[mod_key].obs["id"] = (
            mdata.mod[mod_key].obs["plate_id"].astype(str)
            + "_"
            + mdata.mod[mod_key].obs["well_id"].astype(str)
        )
    sample_ids = mdata.mod["phenocoder"].obs["id"].unique()
    # list of batches sample_ids
    batch_sample_ids = [
        sample_ids[i : i + args.batch_size]
        for i in range(0, len(sample_ids), args.batch_size)
    ]
    assert len(batch_sample_ids) >= args.batch

    # run benchmarks
    benchmarking.run_benchmark_dataset(
        mdata,
        sample_ids=batch_sample_ids[args.batch].tolist(),
        batch_id=args.batch,
        n_neighbors=15,
        out_tmp=dir_results,
        mod_keys=mod_keys,
    )

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
    dir_adata = "data/pilotscreen/anndata"
    for cycle in [1, 3]:
        dir_results = Path(dir_adata, "benchmarking", f"cycle_0{cycle}")
        dir_results.mkdir(parents=True, exist_ok=True)

        file = f"{dir_adata}/mdata_cycle-0{cycle}.h5mu"
        # read mdatas
        mdata = mu.read_h5mu(file)
        for mod_key in ["phenocoder", "phenocoder_msg", "nuclei", "nuclei_msg"]:
            mdata.mod[mod_key].obs["id"] = (
                mdata.mod[mod_key].obs["plate_id"].astype(str) + "_" + mdata.mod[mod_key].obs["well_id"].astype(str)
            )
        sample_ids = mdata.mod["phenocoder"].obs["id"].unique()
        # list of batches sample_ids
        batch_sample_ids = [sample_ids[i : i + args.batch_size] for i in range(0, len(sample_ids), args.batch_size)]
        assert len(batch_sample_ids) >= args.batch

        # run benchmarks
        benchmarking.run_benchmark_dataset(
            mdata,
            sample_ids=batch_sample_ids[args.batch].tolist(),
            batch_id=args.batch,
            out_tmp=dir_results,
        )

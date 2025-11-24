import os
import pickle
import random
from pathlib import Path

import anndata as ad
import marsilea as ma
import marsilea.plotter as mp
import matplotlib.pyplot as plt
import muon as mu
import networkx as nx
import numpy as np
import pandas as pd
import pynndescent
import scanpy as sc
import seaborn as sns
import yaml
from matplotlib import colors
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score

from whole_mount_tumoroid.phenocoder.cluster import run_clustering
from whole_mount_tumoroid.phenocoder.utils import load_plate


def expand_channels_by_set(
    adata: ad.AnnData, staining_sets, channel_cols
) -> ad.AnnData:
    adata_exp = adata.copy()
    for staining_set in staining_sets:
        for channel in channel_cols:
            new_col_name = f"{staining_set}_{channel}"
            adata_exp.obs[new_col_name] = np.nan

    for staining_set in staining_sets:
        mask = adata_exp.obs["staining_set"] == staining_set
        for channel in channel_cols:
            new_col_name = f"{staining_set}_{channel}"
            adata_exp.obs.loc[mask, new_col_name] = adata_exp.obs.loc[mask, channel]

    adata_exp.obs.drop(columns=channel_cols, inplace=True)
    return adata_exp


def expand_channels_by_stain(adata: ad.AnnData, stain_dict, channel_cols) -> ad.AnnData:
    adata_exp = adata.copy()
    for stain in stain_dict:
        adata_exp.obs[stain] = np.nan

    for stain in stain_dict:
        for stain_channel_pair in stain_dict[stain]:
            staining_set = stain_channel_pair[0]
            channel = stain_channel_pair[1]
            channel_col = [c for c in channel_cols if str(channel) in c][0]
            mask = adata_exp.obs["staining_set"] == staining_set
            adata_exp.obs.loc[mask, stain] = adata_exp.obs.loc[mask, channel_col]

    adata_exp.obs.drop(columns=channel_cols, inplace=True)
    return adata_exp


def subset_adata(adata: ad.AnnData, staining_sets, n_cells: int = 1000) -> ad.AnnData:
    selected_indices = []
    for staining_set in staining_sets:
        staining_set_mask = adata.obs["staining_set"] == staining_set
        mask_index = np.where(staining_set_mask)[0]

        sampled_indices = np.random.choice(mask_index, size=n_cells, replace=False)
        selected_indices.extend(sampled_indices)

    adata_subset = adata[selected_indices].copy()

    print(adata.shape)
    print(adata_subset.shape)
    return adata_subset


def plot_large_adjacency_heatmap(
    adjacency_matrix, max_size=1200, title="Adjacency Matrix"
):
    """
    Plot large adjacency matrix by downsampling.
    """
    if hasattr(adjacency_matrix, "toarray"):
        adj_dense = adjacency_matrix.toarray()
    else:
        adj_dense = adjacency_matrix

    n = adj_dense.shape[0]

    if n > max_size:
        # Downsample by taking every nth element
        step = n // max_size
        indices = np.arange(0, n, step)[:max_size]
        adj_subset = adj_dense[np.ix_(indices, indices)]
        print(
            f"Downsampled from {n}x{n} to {adj_subset.shape[0]}x{adj_subset.shape[1]}"
        )
    else:
        adj_subset = adj_dense
        indices = np.arange(n)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        adj_subset,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title(f"{title} (downsampled)" if n > max_size else title)
    plt.tight_layout()
    plt.show()


def calc_adjancency_matrix(adata: ad.AnnData, mask, n_neighbors=3, metric="euclidean"):
    group_indices = np.where(mask)[0]
    subset_pca = adata.obsm["X_pca"][mask]
    index = pynndescent.NNDescent(subset_pca, metric=metric)

    neighbor_indices, distances = index.query(
        adata.obsm["X_pca"], k=n_neighbors + 1
    )  # plus one so we can remove the closest "neighbor" of the observation later, as it is almost always the obs itself

    # calc search tree accuracy before removing self-first neighbors
    group_neighbors = neighbor_indices[mask]
    self_as_first = group_neighbors[:, 0] == np.arange(len(group_indices))
    accuracy = np.mean(self_as_first)
    print(f"Self-neighbor accuracy before filtering: {accuracy:.5%}")
    # Skip the first neighbor (self) for each query
    neighbor_indices = neighbor_indices[:, 1:]  # Remove first column

    group_neighbors = neighbor_indices[mask]
    self_as_first = group_neighbors[:, 0] == np.arange(len(group_indices))
    accuracy = np.mean(self_as_first)
    print(f"Self-neighbor accuracy after filtering: {accuracy:.5%}")

    global_neighbor_indices = group_indices[neighbor_indices]

    # Build sparse adjacency matrix
    n_obs = adata.n_obs
    row_indices = []
    col_indices = []

    for i in range(n_obs):
        for j in range(n_neighbors):
            row_indices.append(i)
            col_indices.append(global_neighbor_indices[i, j])

    # Create adjacency matrix (binary: 1 if connected, 0 otherwise)
    A = csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)), shape=(n_obs, n_obs)
    )

    # plot_large_adjacency_heatmap(A, max_size=10000)

    return A


def build_matrix_dict(
    adata: ad.AnnData,
    staining_sets: list,
    stain_dict=None,
    n_neighbors=3,
    metric="euclidean",
    by_timepoints: bool = False,
    cross_val_mask_dict=None,
):
    adjacency_matrices = {}

    plates = sorted(adata.obs["plate_id"].unique())
    if stain_dict:
        for stain in stain_dict:
            print(stain)
            mask = adata.obs["staining_set"].isin([x[0] for x in stain_dict[stain]])
            if cross_val_mask_dict is not None:
                mask = mask & ~cross_val_mask_dict[stain].reindex(
                    mask.index, fill_value=False
                )
            adjacency_matrices[stain] = calc_adjancency_matrix(
                adata, mask, n_neighbors, metric
            )
    else:
        for staining_set in staining_sets:
            if by_timepoints:
                for plate_id in plates:
                    print(
                        f"Finding k-nearest neighbors for set {staining_set} in plate {plate_id} "
                    )
                    matrix_id = f"{staining_set}_{plate_id}"
                    mask = (adata.obs["staining_set"] == staining_set) & (
                        adata.obs["plate_id"] == plate_id
                    )
                    if cross_val_mask_dict is not None:
                        mask = mask & ~cross_val_mask_dict[stain].reindex(
                            mask.index, fill_value=False
                        )
                    adjacency_matrices[matrix_id] = calc_adjancency_matrix(
                        adata, mask, n_neighbors, metric
                    )
            else:
                print(f"Finding k-nearest neighbors for set {staining_set}")
                mask = adata.obs["staining_set"] == staining_set
                if cross_val_mask_dict is not None:
                    mask = mask & ~cross_val_mask_dict[stain].reindex(
                        mask.index, fill_value=False
                    )
                adjacency_matrices[staining_set] = calc_adjancency_matrix(
                    adata, mask, n_neighbors, metric
                )

    return adjacency_matrices


def impute_staining_sets(
    adata: ad.AnnData,
    staining_sets,
    matrix_dict,
    n_neighbors=3,
    by_timepoints: bool = False,
) -> ad.AnnData:
    imputed_features = []
    var_names = []
    metric_df = pd.DataFrame()
    plates = sorted(adata.obs["plate_id"].unique())

    for target_staining_set in staining_sets:
        print(f"Imputing stainings for {target_staining_set}...")
        impute_columns = [
            col
            for col in adata.obs.columns
            if col.startswith(f"{target_staining_set}_ch")
        ]
        feature_df = adata.obs[impute_columns].copy()

        adata_tmp = ad.AnnData(feature_df)
        adata_tmp.obs["z"] = adata.obs["z"].values
        adata_tmp.obs["plate_id"] = adata.obs["plate_id"].values

        set_mask = adata.obs["staining_set"] == target_staining_set
        adata_tmp_sub = adata_tmp[set_mask, :]

        sc.pp.regress_out(adata_tmp_sub, ["z"])
        sc.pp.scale(adata_tmp_sub)
        adata_tmp.X[set_mask, :] = adata_tmp_sub.X

        adata_tmp.layers["imputed"] = (
            adata_tmp.X.copy()
        )  # placeholder to initiate the layer

        if not by_timepoints:
            print(f"Imputing {target_staining_set}")
            A = matrix_dict[target_staining_set]
            # plot_large_adjacency_heatmap(A, max_size=10000)

            imputed = (
                A.dot(adata_tmp.X) / n_neighbors
            )  # divide the dot product sum by nieghbors to get mean

            adata_tmp.layers["imputed"] = imputed
        else:
            print(f"Imputing {target_staining_set}")
            for plate_id in plates:
                print(f"Imputing plate {plate_id}")
                A = matrix_dict[f"{target_staining_set}_{plate_id}"]
                # plot_large_adjacency_heatmap(A, max_size=10000)
                imputed = (
                    A.dot(adata_tmp.X) / n_neighbors
                )  # divide the dot product sum by nieghbors to get mean

                timepoint_mask = adata_tmp.obs["plate_id"] == plate_id
                adata_tmp.layers["imputed"][timepoint_mask, :] = imputed[
                    timepoint_mask, :
                ]

        for plate_id in plates:
            new_mask = (set_mask) & (adata_tmp.obs["plate_id"] == plate_id)

            ref_subset = adata_tmp.X[new_mask].reshape(-1, len(impute_columns))
            imputed_subset = adata_tmp.layers["imputed"][new_mask].reshape(
                -1, len(impute_columns)
            )

            for i, column in enumerate(impute_columns):
                euclidean_dist = np.linalg.norm(ref_subset[:, i] - imputed_subset[:, i])
                msd = np.mean((ref_subset[:, i] - imputed_subset[:, i]) ** 2)
                rmsd = np.sqrt(msd)
                pearson_corr = pearsonr(ref_subset[:, i], imputed_subset[:, i])[0]
                spearman_corr = spearmanr(ref_subset[:, i], imputed_subset[:, i])[0]
                r2 = r2_score(ref_subset[:, i], imputed_subset[:, i])
                metric_df.loc[column, f"{plate_id}_euclidean_dist"] = euclidean_dist
                metric_df.loc[column, f"{plate_id}_msd"] = msd
                metric_df.loc[column, f"{plate_id}_rmsd"] = rmsd
                metric_df.loc[column, f"{plate_id}_pearson_corr"] = pearson_corr
                metric_df.loc[column, f"{plate_id}_spearman_corr"] = spearman_corr
                metric_df.loc[column, f"{plate_id}_r2"] = r2
                metric_df.loc[column, "staining_set"] = target_staining_set
                metric_df.loc[column, "channel"] = int(i + 2)

        imputed_features.append(adata_tmp.layers["imputed"])
        var_names.extend(impute_columns)
        adata.obs = adata.obs.drop(impute_columns, axis=1)

    imputed_features_X = np.hstack(imputed_features)
    adata_result = ad.AnnData(imputed_features_X, obs=adata.obs, var=metric_df)

    return adata_result


def impute_staining_stains(
    adata: ad.AnnData,
    stain_dict,
    matrix_dict,
    n_neighbors=3,
    by_timepoints: bool = False,
    cross_val_mask_dict=None,
) -> ad.AnnData:
    imputed_features = []
    var_names = []
    metric_df = pd.DataFrame()
    cv_metric_df = pd.DataFrame()
    plates = sorted(adata.obs["plate_id"].unique())

    for stain in stain_dict:
        print(f"Imputing stainings for {stain}...")
        impute_columns = [
            col for col in adata.obs.columns if col.startswith(f"{stain}")
        ]
        feature_df = adata.obs[impute_columns].copy()

        adata_tmp = ad.AnnData(feature_df)
        adata_tmp.obs["z"] = adata.obs["z"].values
        adata_tmp.obs["plate_id"] = adata.obs["plate_id"].values
        adata_tmp.obs["well_id"] = adata.obs["well_id"].values
        sets_with_stain = [x[0] for x in stain_dict[stain]]

        set_mask = adata.obs["staining_set"].isin(sets_with_stain)
        adata_tmp_sub = adata_tmp[set_mask, :]

        sc.pp.regress_out(adata_tmp_sub, ["z"])
        sc.pp.scale(adata_tmp_sub)
        adata_tmp.X[set_mask, :] = adata_tmp_sub.X

        # Store original values for CV comparison
        original_X = adata_tmp.X.copy()

        adata_tmp.layers["imputed"] = (
            adata_tmp.X.copy()
        )  # placeholder to initiate the layer

        print(f"Imputing {stain}")
        A = matrix_dict[stain]
        # plot_large_adjacency_heatmap(A, max_size=10000)

        imputed = (
            A.dot(adata_tmp.X) / n_neighbors
        )  # divide the dot product sum by nieghbors to get mean

        adata_tmp.layers["imputed"] = imputed

        for plate_id in plates:
            new_mask = (set_mask) & (adata_tmp.obs["plate_id"] == plate_id)

            ref_subset = adata_tmp.X[new_mask].reshape(-1, len(impute_columns))
            imputed_subset = adata_tmp.layers["imputed"][new_mask].reshape(
                -1, len(impute_columns)
            )

            for i, column in enumerate(impute_columns):
                euclidean_dist = np.linalg.norm(ref_subset[:, i] - imputed_subset[:, i])
                msd = np.mean((ref_subset[:, i] - imputed_subset[:, i]) ** 2)
                rmsd = np.sqrt(msd)
                pearson_corr = pearsonr(ref_subset[:, i], imputed_subset[:, i])[0]
                spearman_corr = spearmanr(ref_subset[:, i], imputed_subset[:, i])[0]
                r2 = r2_score(ref_subset[:, i], imputed_subset[:, i])
                metric_df.loc[column, f"{plate_id}_euclidean_dist"] = euclidean_dist
                metric_df.loc[column, f"{plate_id}_msd"] = msd
                metric_df.loc[column, f"{plate_id}_rmsd"] = rmsd
                metric_df.loc[column, f"{plate_id}_pearson_corr"] = pearson_corr
                metric_df.loc[column, f"{plate_id}_spearman_corr"] = spearman_corr
                metric_df.loc[column, f"{plate_id}_r2"] = r2
                metric_df.loc[column, "stain"] = stain
                metric_df.loc[column, "channel"] = int(i + 2)

            # Cross-validation metrics (new code)
            if cross_val_mask_dict and stain in cross_val_mask_dict:
                print(f"Computing cross-validation metrics for {stain}...")
                cv_mask = cross_val_mask_dict[stain]

                cv_wells_mask = new_mask & cv_mask  # Only cells in this stain's sets

                cv_original = original_X[cv_wells_mask].reshape(-1, len(impute_columns))
                cv_imputed = adata_tmp.layers["imputed"][cv_wells_mask].reshape(
                    -1, len(impute_columns)
                )

                print(f"Cross-validation on {cv_wells_mask.sum()} held-out cells")

                for i, column in enumerate(impute_columns):
                    cv_euclidean_dist = np.linalg.norm(
                        cv_original[:, i] - cv_imputed[:, i]
                    )
                    cv_msd = np.mean((cv_original[:, i] - cv_imputed[:, i]) ** 2)
                    cv_rmsd = np.sqrt(cv_msd)
                    cv_pearson_corr = pearsonr(cv_original[:, i], cv_imputed[:, i])[0]
                    cv_spearman_corr = spearmanr(cv_original[:, i], cv_imputed[:, i])[0]
                    cv_r2 = r2_score(cv_original[:, i], cv_imputed[:, i])

                    cv_metric_df.loc[column, f"{plate_id}_cv_euclidean_dist"] = (
                        cv_euclidean_dist
                    )
                    cv_metric_df.loc[column, f"{plate_id}_cv_msd"] = cv_msd
                    cv_metric_df.loc[column, f"{plate_id}_cv_rmsd"] = cv_rmsd
                    cv_metric_df.loc[column, f"{plate_id}_cv_pearson_corr"] = (
                        cv_pearson_corr
                    )
                    cv_metric_df.loc[column, f"{plate_id}_cv_spearman_corr"] = (
                        cv_spearman_corr
                    )
                    cv_metric_df.loc[column, f"{plate_id}_cv_r2"] = cv_r2
                    cv_metric_df.loc[column, f"{plate_id}_cv_n_cells"] = (
                        cv_wells_mask.sum()
                    )

        imputed_features.append(adata_tmp.layers["imputed"])
        var_names.extend(impute_columns)
        adata.obs = adata.obs.drop(impute_columns, axis=1)

    if not cv_metric_df.empty:
        combined_metrics = metric_df.join(cv_metric_df, how="outer")
    else:
        combined_metrics = metric_df

    imputed_features_X = np.hstack(imputed_features)
    adata_result = ad.AnnData(imputed_features_X, obs=adata.obs, var=combined_metrics)

    return adata_result


if __name__ == "__main__":
    screen = "timecourse"
    file = "whole_mount_tumoroid/configs/params.yaml"

    stain_metadata = pd.read_csv("metafiles/timecourse_stainings_metadata.csv")
    stain_metadata["staining_set"] = stain_metadata["staining_set"].astype(str)
    stains = stain_metadata.stain.unique()
    stain_dict = (
        stain_metadata.groupby("stain")[["staining_set", "channel"]]
        .apply(lambda x: x.values.tolist())
        .to_dict()
    )

    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    dir_results = Path(params["dir_screen"], "anndata")

    df_plate_layout = pd.read_csv(Path(params["dir_screen"], "timecourse_layout.csv"))
    df_plate_layout = df_plate_layout.melt(
        id_vars=["row"], var_name="col", value_name="staining_set"
    )
    df_plate_layout["column"] = df_plate_layout["col"].str.zfill(2)
    df_plate_layout["well_id"] = df_plate_layout["row"] + df_plate_layout["column"]

    dir_screen = params["dir_screen"]
    plates = params["plates"]
    df_plates = pd.concat(
        [
            pd.read_csv(
                f"{dir_screen}/{plate}/plate_information.csv", dtype={plate: str}
            ).assign(plate=plate)
            for plate in plates
        ]
    )

    df_plate_layouts = df_plate_layout.merge(df_plates, how="cross")
    df_plate_layouts["plate_id"] = df_plate_layouts["plate"]
    df_stain_layout = df_plate_layouts[["well_id", "plate_id", "staining_set"]]
    df_stain_layout = df_stain_layout.map(str)

    # get nuclei channel columns
    plates = params["plates"]
    df = []
    for plate in plates:
        df_plate = load_plate(
            plate,
            params["input_type"],
            params["dir_screen"],
            params["registered"],
            z_step=10,
            plate_id=f"{plate}-01",
        )
        df.append(df_plate)
    df = pd.concat(df)
    df["label"] = df["label"].astype(str) + "_" + df["well"] + "_" + df["plate"]

    # load phenocoder data with image patch embedding PCA space
    mdata = mu.read_h5mu(Path(dir_results, "mdata_registered.h5mu"))
    mod = "phenocoder"
    adata = mdata[mod].copy()

    imputation = "neighbors"
    by_timepoints = False
    imp_mode = f"{imputation}_bytimepoints_{by_timepoints}"

    adata.obs = adata.obs.reset_index().merge(
        df_stain_layout, how="left", on=["well_id", "plate_id"]
    )
    adata.obs = adata.obs.merge(
        df[
            [
                "label",
                f"ch_02_{imputation}",
                f"ch_03_{imputation}",
                f"ch_04_{imputation}",
            ]
        ],
        how="left",
        on=["label"],
    )

    staining_sets = sorted(adata.obs["staining_set"].unique())
    n_neighbors = 3
    metric = "euclidean"

    # channels to impute for each staining set
    channel_cols = [f"ch_02_{imputation}", f"ch_03_{imputation}", f"ch_04_{imputation}"]

    n_runs = 1
    metric_df_list = []

    adata_sub = subset_adata(adata, staining_sets=staining_sets, n_cells=10000)

    for run_id in range(n_runs):
        print(f"RUN NUMBER {run_id}")

        # adata_in = adata.copy()
        adata_in = adata_sub.copy()

        mask_dict = {}
        for stain in stain_dict:
            sampled_wells = []
            for set in stain_dict[stain]:
                stain_wells = list(
                    adata_in[adata_in.obs["staining_set"] == set[0]]
                    .obs["well_id"]
                    .unique()
                )
                sampled_wells.extend(
                    random.sample(stain_wells, 2)
                )  # sample 25% (2 of 8 wells per staining set)
            cross_val_mask = adata_in.obs["well_id"].isin(sampled_wells)
            mask_dict[stain] = cross_val_mask

        # create a column for each staining set + channel combination
        # adata_exp_1 = expand_channels_by_set(adata, staining_sets, channel_cols)
        adata_exp_2 = expand_channels_by_stain(adata_in, stain_dict, channel_cols)

        # Build search trees for each staining set
        # adjacency_matrix_dict_1 = build_matrix_dict(adata_exp_1, staining_sets, n_neighbors=n_neighbors, metric=metric, by_timepoints=by_timepoints)
        adjacency_matrix_dict_2 = build_matrix_dict(
            adata_exp_2,
            staining_sets,
            stain_dict=stain_dict,
            n_neighbors=n_neighbors,
            metric=metric,
            by_timepoints=by_timepoints,
            cross_val_mask_dict=mask_dict,
        )

        # Impute stainings
        adata_copy = adata.copy()
        # adata_out_1 = impute_staining_sets(adata_exp_1, staining_sets, adjacency_matrix_dict_1, n_neighbors=n_neighbors, by_timepoints=by_timepoints)
        adata_out_2 = impute_staining_stains(
            adata_exp_2,
            stain_dict,
            adjacency_matrix_dict_2,
            n_neighbors=n_neighbors,
            by_timepoints=by_timepoints,
            cross_val_mask_dict=mask_dict,
        )

        metric_df = adata_out_2.var.copy()
        metric_df["run"] = run_id
        metric_df_list.append(metric_df)

    # Combine all runs into a single dataframe
    all_runs_df = pd.concat(metric_df_list, ignore_index=True)

    all_runs_df.to_pickle("all_runs_neighbors_df_10_9.pkl")

    # Calculate summary statistics with error bars
    summary_stats = []

    for run in range(n_runs):
        if pd.api.types.is_numeric_dtype(all_runs_df[metric_col]):
            stats = {
                "metric": metric_col,
                "mean": all_runs_df[metric_col].mean(),
                "std": all_runs_df[metric_col].std(),
                "sem": all_runs_df[metric_col].sem(),
                "ci_lower": all_runs_df[metric_col].quantile(0.025),
                "ci_upper": all_runs_df[metric_col].quantile(0.975),
                "min": all_runs_df[metric_col].min(),
                "max": all_runs_df[metric_col].max(),
            }
            summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    print("\nSummary statistics across all runs:")
    print(summary_df)

    metrics = [x for x in metric_df_50 if ("cv" in x and "corr" in x)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(metric_df_50[metrics], annot=True, fmt=".2f", cmap="plasma")
    plt.tight_layout()

    metrics = [x for x in metric_df_50 if ("cv" not in x and "corr" in x)]

    plt.figure(figsize=(10, 8))
    sns.heatmap(metric_df_50[metrics], annot=True, fmt=".2f", cmap="plasma")
    plt.tight_layout()

    mdata.mod[f"imputed_{imp_mode}"] = adata_out_2

    # set obs index
    mdata[f"imputed_{imp_mode}"].obs.index = mdata[f"imputed_{imp_mode}"].obs["label"]

    # set proper var names
    index = mdata[f"imputed_{imp_mode}"].var.index
    mdata[f"imputed_{imp_mode}"].var = mdata[f"imputed_{imp_mode}"].var.merge(
        stain_metadata, how="left", on=["staining_set", "channel"]
    )
    mdata[f"imputed_{imp_mode}"].var.index = index
    mdata[f"imputed_{imp_mode}"].var_names = (
        mdata[f"imputed_{imp_mode}"].var["stain"]
        + "_"
        + mdata[f"imputed_{imp_mode}"].var["staining_set"]
    )

    # mdata.write_h5mu(Path(dir_results, 'mdata_registered_imputed.h5mu'))

    mdata = mu.read_h5mu(Path(dir_results, "mdata_registered_imputed.h5mu"))

    mdata.mod[f"imputed_{imp_mode}"] = run_clustering(
        mdata[f"imputed_{imp_mode}"],
        subsampling=False,
        frac=0.1,
        n_comps=params["phenocoder"]["n_comps_pca"]["imputed"],
        resolution=params["phenocoder"]["cluster_res"]["imputed"],
        harmony=False,
        use_gpu=False,
    )
    # var_subset='subset_mask')

    # mdata.write_h5mu(Path(dir_results, 'mdata_registered_imputed.h5mu'))

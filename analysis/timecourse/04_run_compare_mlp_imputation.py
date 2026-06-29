import pickle
from pathlib import Path

import muon as mu
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from analysis.timecourse.mlp_training import (
    ConditionalMarkerMLP,
    MarkerDataset,
    data_setup,
    get_train_val_test_indices,
)
from analysis.timecourse.plot_mlp_imputation import plot_summary
from phenocoder.utils import load_plate


def fit_linear_baseline(
    embeddings, marker_ids, intensities, train_idx, val_idx, marker_names
):
    num_markers = len(marker_names)
    one_hot = np.zeros((len(embeddings), num_markers))
    one_hot[np.arange(len(embeddings)), marker_ids] = 1.0
    X = np.concatenate([embeddings, one_hot], axis=1)

    fit_idx = np.concatenate([train_idx, val_idx])
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X[fit_idx])

    model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], scoring="r2")
    model.fit(X_fit, intensities[fit_idx])
    print(f"Best alpha: {model.alpha_}")
    return model, scaler


def evaluate_linear_baseline(
    model, scaler, embeddings, marker_ids, intensities, test_idx, marker_names
):
    num_markers = len(marker_names)
    one_hot = np.zeros((len(embeddings), num_markers))
    one_hot[np.arange(len(embeddings)), marker_ids] = 1.0
    X_test = scaler.transform(np.concatenate([embeddings, one_hot], axis=1)[test_idx])
    y_test = intensities[test_idx]
    preds = model.predict(X_test)
    marker_ids_test = marker_ids[test_idx]

    metrics = []
    for i, marker in enumerate(marker_names):
        mask = marker_ids_test == i
        if mask.sum() == 0:
            continue
        metrics.append(
            {
                "marker": marker,
                "R²": r2_score(y_test[mask], preds[mask]),
                "Pearson": pearsonr(y_test[mask], preds[mask])[0],
                "Spearman": spearmanr(y_test[mask], preds[mask])[0],
                "MAE": mean_absolute_error(y_test[mask], preds[mask]),
            }
        )

    return pd.DataFrame(metrics).set_index("marker")


@torch.no_grad()
def evaluate(model, val_loader, marker_names, device="cuda"):
    model.eval().to(device)

    all_preds = []
    all_targets = []
    all_markers = []

    for dapi, marker_ids, targets in val_loader:
        dapi, marker_ids, targets = (
            dapi.to(device),
            marker_ids.to(device),
            targets.to(device),
        )
        preds = model(dapi, marker_ids)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_markers.append(marker_ids.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_markers = torch.cat(all_markers).numpy()

    df = pd.DataFrame(
        {
            "marker_id": all_markers,
            "marker": [marker_names[m] for m in all_markers],
            "pred": all_preds,
            "target": all_targets,
        }
    )

    return df


def compute_metrics(df, marker_names):
    metrics = []
    for marker in marker_names:
        sub = df[df["marker"] == marker]
        pred, target = sub["pred"].values, sub["target"].values
        if len(pred) < 2:
            continue
        metrics.append(
            {
                "marker": marker,
                "R²": r2_score(target, pred),
                "Pearson": pearsonr(target, pred)[0],
                "Spearman": spearmanr(target, pred)[0],
                "MAE": mean_absolute_error(target, pred),
            }
        )
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    screen = "timecourse"
    file = "/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/configs/params.yaml"

    stain_metadata = pd.read_csv(
        "/pstore/data/ihb-g-deco/USERS/schulzp9/tumoroid/metafiles/timecourse_stainings_metadata.csv"
    )
    stain_metadata["staining_set"] = stain_metadata["staining_set"].astype(str)
    stains = stain_metadata.stain.unique()
    stain_dict = {
        set_id: dict(group[["channel", "stain"]].values)
        for set_id, group in stain_metadata.groupby("staining_set")
    }

    with open(file) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
        params = params[screen]

    dir_results = Path(params["dir_screen"], "anndata")
    mlp_dir = Path(params["dir_screen"], "mlp_imputation")

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
    df_plates = []
    for plate in plates:
        df_plate = load_plate(
            plate,
            params["input_type"],
            params["dir_screen"],
            params["registered"],
            z_step=10,
            plate_id=f"{plate}-01",
        )
        df_plates.append(df_plate)
    df_plates = pd.concat(df_plates)
    df_plates["label"] = (
        df_plates["label"].astype(str)
        + "_"
        + df_plates["well"]
        + "_"
        + df_plates["plate"]
    )

    # load phenocoder data with image patch embedding PCA space
    mdata = mu.read_h5mu(Path(dir_results, "mdata_registered.h5mu"))

    imputations = ["nuclei", "neighbors", "nuclei", "neighbors"]
    mods = ["phenocoder", "phenocoder", "phenocoder_msg", "phenocoder_msg"]
    weights = [
        f"{mlp_dir}/phenocoder_nuclei_20260415_174021.pth",
        f"{mlp_dir}/phenocoder_neighbors_20260416_092249.pth",
        f"{mlp_dir}/phenocoder_msg_nuclei_20260408_142714.pth",
        f"{mlp_dir}/phenocoder_msg_neighbors_20260408_145135.pth",
    ]

    mlp_metrics = {}  # {"nuclei": df, "neighbors": df}
    baseline_metrics = {}
    nn_metrics = {}

    for imp, w, mod in zip(imputations, weights, mods):
        adata = mdata[mod].copy()

        print(imp, w)
        embeddings, marker_ids, intensities, _ = data_setup(
            adata, df_stain_layout, df_plates, imp, stain_dict
        )

        DAPI_DIM = embeddings.shape[1]
        NUM_MARKERS = len(stains)

        train_idx, val_idx, test_idx = get_train_val_test_indices(len(embeddings))

        # --- MLP ---
        model = ConditionalMarkerMLP(
            dapi_dim=DAPI_DIM,
            num_markers=NUM_MARKERS,
            hidden_dims=[512, 256, 128],
            dropout=0.1,
        )
        model.load_state_dict(torch.load(w))
        model.eval()

        # --- Ridge ---
        lin_model, scaler = fit_linear_baseline(
            embeddings, marker_ids, intensities, train_idx, val_idx, stains
        )

        # --- split test into groups ---
        test_idx_shuffled = test_idx.copy()
        np.random.shuffle(test_idx_shuffled)
        groups = np.array_split(test_idx_shuffled, 5)

        mlp_group_metrics = []
        baseline_group_metrics = []

        for group in groups:
            # MLP
            test_ds = MarkerDataset(
                embeddings[group], marker_ids[group], intensities[group]
            )
            test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
            df = evaluate(model, test_loader, stains)
            metric_df = compute_metrics(df, stains)
            mlp_group_metrics.append(metric_df)

            # Ridge
            baseline_df = evaluate_linear_baseline(
                lin_model,
                scaler,
                embeddings.numpy(),
                marker_ids.numpy(),
                intensities.numpy(),
                group,
                stains,
            )
            baseline_group_metrics.append(baseline_df)

        # -- Load kNN results ---
        with open(
            f"/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/analysis/timecourse/cv_imputation/new_{mod}_{imp}.pkl",
            "rb",
        ) as f:
            nn_imputation = pickle.load(f)

            cols = [x for x in nn_imputation.columns if "spearman" in x]
            nn_imp = nn_imputation[cols]
            cv_cols = [x for x in nn_imp.columns if "cv" in x]
            nn_imp_cv = nn_imp[cv_cols]
            nn_imp_cv["marker"] = nn_imputation["stain"]

            nn_imp_cv_melt = pd.melt(
                nn_imp_cv,
                id_vars=["marker"],
                var_name="imputation",
                value_name="Spearman",
            ).reset_index(drop=True)

        mlp_metrics[f"{imp}_{mod}"] = pd.concat(mlp_group_metrics).reset_index()
        baseline_metrics[f"{imp}_{mod}"] = pd.concat(
            baseline_group_metrics
        ).reset_index()
        nn_metrics[f"{imp}_{mod}"] = nn_imp_cv_melt.reset_index()

    melt_stack = []
    for i in range(2):
        with open(
            f"/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/analysis/timecourse/cv_imputation/all_runs_df_10_{i}.pkl",
            "rb",
        ) as f:
            nn_imputation = pickle.load(f)

        cols = [x for x in nn_imputation.columns if "spearman" in x]
        nn_imp = nn_imputation[cols]
        cv_cols = [x for x in nn_imp.columns if "cv" in x]
        nn_imp_cv = nn_imp[cv_cols]
        nn_imp_cv["marker"] = nn_imputation["stain"]

        nn_imp_cv_melt = pd.melt(
            nn_imp_cv, id_vars=["marker"], var_name="imputation", value_name="Spearman"
        ).reset_index(drop=True)
        melt_stack.append(nn_imp_cv_melt)

    nn_metrics_old = {"nuclei": pd.concat(melt_stack)}

    melt_stack = []
    for i in range(2):
        with open(
            f"/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/analysis/timecourse/cv_imputation/all_runs_neighbors_df_10_{i}.pkl",
            "rb",
        ) as f:
            nn_imputation = pickle.load(f)

        cols = [x for x in nn_imputation.columns if "spearman" in x]
        nn_imp = nn_imputation[cols]

        cv_cols = [x for x in nn_imp.columns if "cv" in x]
        nn_imp_cv = nn_imp[cv_cols]
        nn_imp_cv["marker"] = nn_imputation["stain"]

        nn_imp_cv_melt = pd.melt(
            nn_imp_cv, id_vars=["marker"], var_name="imputation", value_name="Spearman"
        ).reset_index(drop=True)
        melt_stack.append(nn_imp_cv_melt)

    nn_metrics_old["neighbors"] = pd.concat(melt_stack)

    colors = {
        "kNN Nuclei": "mediumseagreen",
        "kNN Neighbors": "lightgreen",
        "kNN Nuclei msg": "mediumseagreen",
        "kNN Neighbors msg": "lightgreen",
        "Ridge Nuclei": "coral",
        "Ridge Neighbors": "lightsalmon",
        "Ridge Nuclei msg": "coral",
        "Ridge Neighbors msg": "lightsalmon",
        "MLP Nuclei": "steelblue",
        "MLP Neighbors": "cornflowerblue",
        "MLP Nuclei msg": "steelblue",
        "MLP Neighbors msg": "cornflowerblue",
    }

    datasets = {
        "kNN Nuclei": nn_metrics["nuclei_phenocoder"],
        "kNN Neighbors": nn_metrics["neighbors_phenocoder"],
        "kNN Nuclei msg": nn_metrics["nuclei_phenocoder_msg"],
        "kNN Neighbors msg": nn_metrics["neighbors_phenocoder_msg"],
        "Ridge Nuclei": baseline_metrics["nuclei_phenocoder"],
        "Ridge Neighbors": baseline_metrics["neighbors_phenocoder"],
        "Ridge Nuclei msg": baseline_metrics["nuclei_phenocoder_msg"],
        "Ridge Neighbors msg": baseline_metrics["neighbors_phenocoder_msg"],
        "MLP Nuclei": mlp_metrics["nuclei_phenocoder"],
        "MLP Neighbors": mlp_metrics["neighbors_phenocoder"],
        "MLP Nuclei msg": mlp_metrics["nuclei_phenocoder_msg"],
        "MLP Neighbors msg": mlp_metrics["neighbors_phenocoder_msg"],
    }

    plot_summary(
        mlp_metrics,
        baseline_metrics,
        nn_metrics,
        stains,
        colors,
        datasets,
        jitter_points=False,
        save_plot=False,
    )

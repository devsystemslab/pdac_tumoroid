from pathlib import Path

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
from tqdm import tqdm


def undersampling_analysis(csv_path, max_step=10):
    df = pd.read_csv(csv_path)
    # df = df[(df['area']>100)&(df['area']<2200)]
    z_slices = np.sort(df["z_stack"].unique())
    n_total = len(z_slices)
    # log1p for all intensity columns
    intensity_cols = [col for col in df.columns if "intensity_mean" in col]
    df.loc[:, intensity_cols] = np.log1p(df.loc[:, intensity_cols])
    results = []
    results_merged = []
    for step in range(1, min(max_step + 1, n_total + 1)):
        z_subset = z_slices[::step]
        df_sub = df[df["z_stack"].isin(z_subset)].copy()
        df_sub.loc[:, "z"] = df_sub.loc[:, "z_stack"] * 1.5 / 0.322
        df_merged = (
            df_sub.groupby("label")
            .agg(
                n_z_appearances=("z_stack", "size"),
                area_sum=("area", "sum"),
                area_max=("area", "max"),
                area_mean=("area", "mean"),
                eccentricity_mean=("eccentricity", "mean"),
                intensity_mean=("intensity_mean-0", "mean"),
                intensity_mean_1=("intensity_mean-1", "mean"),
                intensity_mean_2=("intensity_mean-2", "mean"),
                intensity_mean_3=("intensity_mean-3", "mean"),
                major_axis_length_mean=("major_axis_length", "mean"),
                minor_axis_length_mean=("minor_axis_length", "mean"),
                x=("centroid-1", "mean"),
                y=("centroid-0", "mean"),
                z=("z", "mean"),
            )
            .reset_index()
        )
        results_merged.append(df_merged.assign(step=step))
        results.append(
            {
                "step": step,
                "n_z_used": len(z_subset),
                "fraction_z": len(z_subset) / n_total,
                "n_cells": df_merged["label"].nunique(),
                "n_z_appearances_mean": df_merged["n_z_appearances"].mean(),
                "n_z_appearances_std": df_merged["n_z_appearances"].std(),
                "intensity_mean": df_merged["intensity_mean"].mean(),
            }
        )

    df_results = pd.DataFrame(results)
    df_merged = pd.concat(results_merged)
    baseline = df_results.loc[df_results["step"] == 1, "n_cells"].iloc[0]
    df_results["fraction_cells"] = df_results["n_cells"] / baseline
    return df_results, df_merged, df


def calculate_spatial_drift(df):
    wells = df["well"].unique()
    steps = df["step"].unique()
    results = []
    for well in tqdm(wells, desc="Calculating spatial drift"):
        df_well = df[df["well"] == well]
        df_baseline = df_well[df_well["step"] == 1].set_index("label")
        for step in steps:
            df_step = df_well[df_well["step"] == step].set_index("label")
            df_drift = df_step.join(
                df_baseline, how="inner", lsuffix="_current", rsuffix="_baseline"
            )
            df_drift["dx"] = df_drift["x_current"] - df_drift["x_baseline"]
            df_drift["dy"] = df_drift["y_current"] - df_drift["y_baseline"]
            df_drift["dz"] = df_drift["z_current"] - df_drift["z_baseline"]
            df_drift["distance_3d"] = np.sqrt(
                df_drift["dx"] ** 2 + df_drift["dy"] ** 2 + df_drift["dz"] ** 2
            )
            df_drift["step"] = step
            results.append(df_drift.reset_index())
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    plt.ion()
    dir_data = Path(
        "data/segmentation_validation_inhibitors/segmentation_validation_inhibitors/004/004-01/segmentation_plots/shared_merged_labels"
    )
    mdata_org = mu.read_h5mu(
        "data/pilotscreen/anndata/mdata_org_combined.h5mu"
    )
    all_results = []
    all_merged = []
    all_features = []

    for csv_file in tqdm(
        sorted(dir_data.glob("segmentation_features_*.csv")),
        desc="Performing undersampling",
    ):
        sample = csv_file.stem.replace("segmentation_features_", "")
        df_results, df_merged, df_features = undersampling_analysis(csv_file)
        df_results["well"] = sample
        df_merged["well"] = sample
        df_baseline = df_results[df_results["step"] == 1]
        if (
            df_baseline["n_z_used"].values[0] > 300
            and df_baseline["n_cells"].values[0] > 5000
        ):
            all_results.append(df_results)
            all_features.append(df_features)
            all_merged.append(df_merged)

    df_all = pd.concat(all_results)
    df_meta = mdata_org["phenocoder_combined"].obs[
        ["well_id", "conc", "timepoint", "compound", "plate_id"]
    ]
    df_meta = df_meta[df_meta["plate_id"] == "004"]
    # if conc == '0_µM' set compound to 'DMSO'
    df_meta.loc[df_meta["conc"] == "0_µM", "compound"] = "DMSO"
    df_meta["condition"] = (
        df_meta["conc"].astype(str)
        + "_"
        + df_meta["timepoint"].astype(str)
        + "_"
        + df_meta["compound"].astype(str)
    )
    df_meta.rename(columns={"well_id": "well"}, inplace=True)
    # merge with mdata['phenocoder_combined'].obs
    df_all = df_all.merge(df_meta, on="well", how="left")
    df_all = df_all[df_all["compound"] == "DMSO"]
    df_all = df_all[df_all["conc"] != "10_µM"]
    df_all_features = pd.concat(all_features)
    df_all_features = (
        df_all_features.groupby("z_stack")
        .agg({"intensity_mean-0": ["mean", "std"]})
        .reset_index()
    )
    df_all_features.columns = ["z_stack", "mean", "std"]
    df_all_merged = pd.concat(all_merged)

    # Calculate spatial drift for all merged data
    df_all_drift = calculate_spatial_drift(df_all_merged)
    df_all_drift = df_all_drift[df_all_drift["distance_3d"] < 100]
    df_all_drift_agg = (
        df_all_drift.groupby("step")["distance_3d"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    # Set PDF font type for editability in Adobe Illustrator
    plt.rcParams["pdf.fonttype"] = 42

    fig, axes = plt.subplots(2, 5, figsize=(30, 10))
    fig.subplots_adjust(wspace=0.35, hspace=0.4)
    axes[0, 0].plot(
        df_all_features["z_stack"] * 1.5,
        df_all_features["mean"],
        "-",
        alpha=0.3,
        color="steelblue",
    )

    # fraction cells ribbon
    df_summary = (
        df_all.groupby("step")["fraction_cells"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    axes[0, 1].fill_between(
        df_summary["step"] * 1.5,
        df_summary["min"],
        df_summary["max"],
        alpha=0.15,
        color="steelblue",
        label="min–max",
    )
    axes[0, 1].fill_between(
        df_summary["step"] * 1.5,
        df_summary["mean"] - df_summary["std"],
        df_summary["mean"] + df_summary["std"],
        alpha=0.3,
        color="steelblue",
        label="mean ± std",
    )
    axes[0, 1].plot(
        df_summary["step"] * 1.5,
        df_summary["mean"],
        "o-",
        color="steelblue",
        label="mean",
    )
    axes[0, 1].axhline(1.0, ls="--", color="gray", alpha=0.5)
    axes[0, 1].legend()
    # set ylim 0-max
    axes[0, 1].set_ylim(0.5, df_summary["max"].max())
    # n_z_appearances ribbon
    df_z_summary = (
        df_all.groupby("step")["n_z_appearances_mean"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    axes[0, 2].fill_between(
        df_z_summary["step"] * 1.5,
        df_z_summary["min"],
        df_z_summary["max"],
        alpha=0.15,
        color="steelblue",
        label="min–max",
    )
    # add ylim 0-max
    axes[0, 2].set_ylim(0, df_z_summary["max"].max())
    axes[0, 2].fill_between(
        df_z_summary["step"] * 1.5,
        df_z_summary["mean"] - df_z_summary["std"],
        df_z_summary["mean"] + df_z_summary["std"],
        alpha=0.3,
        color="steelblue",
        label="mean ± std",
    )
    axes[0, 2].plot(
        df_z_summary["step"] * 1.5,
        df_z_summary["mean"],
        "o-",
        color="steelblue",
        label="mean",
    )
    axes[0, 2].legend()

    # absolute cell numbers ribbon
    df_abs_summary = (
        df_all.groupby("step")["n_cells"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    axes[0, 3].fill_between(
        df_abs_summary["step"] * 1.5,
        df_abs_summary["min"],
        df_abs_summary["max"],
        alpha=0.15,
        color="steelblue",
        label="min–max",
    )
    axes[0, 3].fill_between(
        df_abs_summary["step"] * 1.5,
        df_abs_summary["mean"] - df_abs_summary["std"],
        df_abs_summary["mean"] + df_abs_summary["std"],
        alpha=0.3,
        color="steelblue",
        label="mean ± std",
    )
    axes[0, 3].plot(
        df_abs_summary["step"] * 1.5,
        df_abs_summary["mean"],
        "o-",
        color="steelblue",
        label="mean",
    )
    # set ylim 0-max
    axes[0, 3].set_ylim(0, df_abs_summary["max"].max())
    axes[0, 3].legend()

    # Add vertical line at 10 µm for normal imaging resolution
    for ax in axes[0, 1:]:
        ax.axvline(10, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    for ax in axes[0, :]:
        ax.set_xlabel("Z-slice spacing [µm]")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_xlabel("Z-depth [µm]")
    axes[0, 0].set_ylabel("DAPI intensity [a.u.]")
    axes[0, 0].set_title("Per well (fraction)")
    axes[0, 1].set_ylabel("Fraction of cells retained")
    axes[0, 1].set_title(f"Fraction summary (n={len(all_results)} wells)")
    axes[0, 2].set_ylabel("Mean z-appearances per cell")
    axes[0, 2].set_title(f"Z-appearances (n={len(all_results)} wells)")
    axes[0, 3].set_ylabel("Absolute cell count")
    axes[0, 3].set_title(f"Absolute cells (n={len(all_results)} wells)")

    # Convert drift from pixels to micrometers (pixel size = 0.322 µm)
    pixel_size_um = 0.322
    df_all_drift_agg["mean_um"] = df_all_drift_agg["mean"] * pixel_size_um
    df_all_drift_agg["std_um"] = df_all_drift_agg["std"] * pixel_size_um

    axes[0, 4].fill_between(
        df_all_drift_agg["step"] * 1.5,
        df_all_drift_agg["mean_um"] - df_all_drift_agg["std_um"],
        df_all_drift_agg["mean_um"] + df_all_drift_agg["std_um"],
        alpha=0.3,
        color="steelblue",
        label="mean ± std",
    )
    axes[0, 4].plot(
        df_all_drift_agg["step"] * 1.5,
        df_all_drift_agg["mean_um"],
        "o-",
        color="steelblue",
        label="mean",
    )

    # Add spatial kernel size reference lines
    axes[0, 4].set_ylabel("3D distance drift [µm]")
    axes[0, 4].set_title(f"Spatial drift (n={len(all_results)} wells)")
    axes[0, 4].legend(loc="upper right")

    axes[0, 4].grid(True, alpha=0.3)

    # SECOND ROW: Condition-grouped plots
    # Define colors for conditions
    conditions = df_all["condition"].dropna().unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(conditions)))
    condition_colors = {cond: colors[i] for i, cond in enumerate(sorted(conditions))}

    # Fraction cells by condition
    for condition in sorted(conditions):
        df_cond = df_all[df_all["condition"] == condition]
        df_cond_summary = (
            df_cond.groupby("step")["fraction_cells"].agg(["mean", "std"]).reset_index()
        )
        axes[1, 1].plot(
            df_cond_summary["step"] * 1.5,
            df_cond_summary["mean"],
            "o-",
            color=condition_colors[condition],
            label=condition,
            alpha=0.7,
        )
    axes[1, 1].axhline(1.0, ls="--", color="gray", alpha=0.5)
    axes[1, 1].set_xlabel("Z-slice spacing [µm]")
    axes[1, 1].set_ylabel("Fraction of cells retained")
    axes[1, 1].set_title("Fraction by condition")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(10, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # n_z_appearances by condition
    for condition in sorted(conditions):
        df_cond = df_all[df_all["condition"] == condition]
        df_cond_summary = (
            df_cond.groupby("step")["n_z_appearances_mean"]
            .agg(["mean", "std"])
            .reset_index()
        )
        axes[1, 2].plot(
            df_cond_summary["step"] * 1.5,
            df_cond_summary["mean"],
            "o-",
            color=condition_colors[condition],
            label=condition,
            alpha=0.7,
        )
    axes[1, 2].set_xlabel("Z-slice spacing [µm]")
    axes[1, 2].set_ylabel("Mean z-appearances per cell")
    axes[1, 2].set_title("Z-appearances by condition")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axvline(10, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # absolute cell numbers by condition
    for condition in sorted(conditions):
        df_cond = df_all[df_all["condition"] == condition]
        df_cond_summary = (
            df_cond.groupby("step")["n_cells"].agg(["mean", "std"]).reset_index()
        )
        axes[1, 3].plot(
            df_cond_summary["step"] * 1.5,
            df_cond_summary["mean"],
            "o-",
            color=condition_colors[condition],
            label=condition,
            alpha=0.7,
        )
    axes[1, 3].set_xlabel("Z-slice spacing [µm]")
    axes[1, 3].set_ylabel("Absolute cell count")
    axes[1, 3].set_title("Absolute cells by condition")
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].axvline(10, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # Spatial drift by condition
    for condition in sorted(conditions):
        df_cond_drift = df_all_drift[
            df_all_drift["well_current"].isin(
                df_all[df_all["condition"] == condition]["well"].unique()
            )
        ]
        df_cond_drift_agg = (
            df_cond_drift.groupby("step")["distance_3d"]
            .agg(["mean", "std"])
            .reset_index()
        )
        # Convert to micrometers
        df_cond_drift_agg["mean_um"] = df_cond_drift_agg["mean"] * pixel_size_um
        df_cond_drift_agg["std_um"] = df_cond_drift_agg["std"] * pixel_size_um
        axes[1, 4].plot(
            df_cond_drift_agg["step"] * 1.5,
            df_cond_drift_agg["mean_um"],
            "o-",
            color=condition_colors[condition],
            label=condition,
            alpha=0.7,
        )
    axes[1, 4].set_xlabel("Z-slice spacing [µm]")
    axes[1, 4].set_ylabel("3D distance drift [µm]")
    axes[1, 4].set_title("Spatial drift by condition")
    axes[1, 4].grid(True, alpha=0.3)
    axes[1, 4].axvline(10, linestyle="--", color="gray", alpha=0.5, linewidth=1)

    # Create consolidated legend in first column of second row
    handles, labels = axes[1, 1].get_legend_handles_labels()
    axes[1, 0].legend(handles, labels, fontsize=7, loc="center", frameon=True)
    axes[1, 0].set_title("Conditions", fontweight="bold")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])

    # plt.tight_layout()
    plt.savefig(dir_data / "undersampling_curve.pdf", dpi=72)
    plt.show()

    # FEATURE DISTRIBUTION PLOTS
    fig_dist, axes_dist = plt.subplots(2, 4, figsize=(28, 10))
    fig_dist.subplots_adjust(wspace=0.3, hspace=0.3)
    axes_dist = axes_dist.flatten()

    # Prepare data for boxplots
    features_to_plot = [
        "intensity_mean",
        "intensity_mean_1",
        "intensity_mean_2",
        "intensity_mean_3",
        "area_mean",
        "major_axis_length_mean",
        "minor_axis_length_mean",
        "eccentricity_mean",
    ]
    feature_labels = {
        "intensity_mean": "Intensity Ch0 [a.u.]",
        "intensity_mean_1": "Intensity Ch1 [a.u.]",
        "intensity_mean_2": "Intensity Ch2 [a.u.]",
        "intensity_mean_3": "Intensity Ch3 [a.u.]",
        "area_mean": "Area [micrometer^2]",
        "major_axis_length_mean": "Major Axis Length [micrometer]",
        "minor_axis_length_mean": "Minor Axis Length [micrometer]",
        "eccentricity_mean": "Eccentricity",
    }

    # Pixel size conversion constant
    pixel_size_um = 0.322

    for idx, feature in enumerate(features_to_plot):
        ax = axes_dist[idx]

        # Get unique steps and sort them
        steps_list = sorted(df_all_merged["step"].unique())

        # Prepare data for violin plot
        plot_data = []
        plot_labels = []
        for step in steps_list:
            df_step = df_all_merged[df_all_merged["step"] == step][feature].copy()
            # Remove NaN values
            df_step = df_step.dropna()
            # Convert pixel-based features to micrometers
            if feature == "area_mean":
                df_step = df_step * (pixel_size_um**2)
            elif feature in ["major_axis_length_mean", "minor_axis_length_mean"]:
                df_step = df_step * pixel_size_um
            # Filter to 5-95 quantiles
            q5 = df_step.quantile(0.05)
            q95 = df_step.quantile(0.95)
            df_step = df_step[(df_step >= q5) & (df_step <= q95)]
            plot_data.append(df_step.values)
            plot_labels.append(f"{step * 1.5:.1f} µm")

        # Create boxplot
        bp = ax.boxplot(
            plot_data,
            positions=range(len(steps_list)),
            widths=0.5,
            patch_artist=True,
            showfliers=False,
        )

        # Customize boxplot colors
        for box in bp["boxes"]:
            box.set_facecolor("steelblue")
            box.set_alpha(0.3)
            box.set_edgecolor("steelblue")

        for whisker in bp["whiskers"]:
            whisker.set_color("steelblue")

        for cap in bp["caps"]:
            cap.set_color("steelblue")

        ax.set_xticks(range(len(steps_list)))
        ax.set_xticklabels(plot_labels, rotation=45)
        ax.set_xlabel("Z-slice spacing")
        ax.set_ylabel(feature_labels.get(feature, feature))
        ax.set_title(f"Distribution of {feature_labels.get(feature, feature)}")
        ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(dir_data / "feature_distributions.pdf", dpi=150)
    plt.show()

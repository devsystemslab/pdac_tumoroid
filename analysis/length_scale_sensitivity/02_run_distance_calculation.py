from pathlib import Path
from typing import Dict

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cosine
from tqdm import tqdm


def calculate_kernel_distances(adata: ad.AnnData, distance_metric: str = "euclidean") -> Dict[int, np.ndarray]:
    """
    Calculate distances between each kernel size representation and the baseline nuclei representation.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object containing message-passing layers for different kernel radii.
        Must have layers named "message_passing_radius_{radius}" for each radius analyzed.
    distance_metric : str, optional
        Distance metric to use. Options: "euclidean", "cosine". Default is "euclidean".

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary where keys are kernel radii and values are arrays of distances
        for each nucleus (n_obs,). Distance is measured between the baseline representation
        (adata.X) and the message-passing aggregated representation at each radius.

    Examples
    --------
    >>> distances = calculate_kernel_distances(adata, distance_metric="euclidean")
    >>> print(distances[5])  # distances for radius=5
    """
    # Get baseline nuclei representation
    baseline = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X

    # Extract all message-passing layers
    radii = []
    layers_dict = {}

    for layer_name in adata.layers.keys():
        if layer_name.startswith("message_passing_radius_"):
            radius = int(layer_name.replace("message_passing_radius_", ""))
            radii.append(radius)
            layers_dict[radius] = adata.layers[layer_name]

    # Sort radii for consistent ordering
    radii = sorted(radii)

    # Calculate distances for each radius
    distances = {}

    for radius in radii:
        kernel_representation = layers_dict[radius]
        kernel_representation = (
            kernel_representation.toarray() if hasattr(kernel_representation, "toarray") else kernel_representation
        )

        # Calculate distance for each nucleus
        nucleus_distances = np.zeros(baseline.shape[0])

        if distance_metric == "euclidean":
            # Vectorized Euclidean distance
            nucleus_distances = np.linalg.norm(baseline - kernel_representation, axis=1)

        elif distance_metric == "cosine":
            # Cosine distance for each row
            for i in range(baseline.shape[0]):
                nucleus_distances[i] = cosine(baseline[i], kernel_representation[i])

        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        distances[radius] = nucleus_distances

    return distances


def calculate_kernel_distances_for_well(
    well_id: str,
    plate_id: str,
    dir_results: str = "data/pilotscreen/anndata/sensitivity_analysis_results",
) -> Dict[int, np.ndarray]:
    """
    Load the sensitivity analysis AnnData for one well and calculate kernel distances.

    Parameters
    ----------
    well_id : str
        Well identifier (e.g., "A1")
    plate_id : str
        Plate identifier (e.g., "plate_001")
    dir_results : str, optional
        Directory containing sensitivity analysis results. Default is the standard path.

    Returns
    -------
    Dict[int, np.ndarray]
        Dictionary where keys are kernel radii and values are arrays of distances
        for each nucleus in this well.

    Raises
    ------
    FileNotFoundError
        If the sensitivity analysis file for the specified well does not exist.

    Examples
    --------
    >>> distances = calculate_kernel_distances_for_well("A1", "plate_001")
    >>> print(f"Radius 5: mean distance = {distances[5].mean():.4f}")
    """
    # Construct file path
    file_path = Path(dir_results) / f"{well_id}_{plate_id}_sensitivity_analysis.h5ad"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Sensitivity analysis file not found: {file_path}\n"
            f"Make sure to run the sensitivity analysis first with run_sensitivity_analysis.py"
        )

    # Load the AnnData object
    adata = ad.read_h5ad(file_path)

    # Calculate distances
    distances = calculate_kernel_distances(adata, distance_metric="euclidean")

    return distances


def calculate_kernel_distances_all_wells(
    dir_results: str = "data/pilotscreen/anndata/sensitivity_analysis_results",
    output_dir: str = None,
    n: int = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Calculate kernel distances for all wells in the sensitivity analysis dataset.

    This is the top-level wrapper that processes all sensitivity analysis results
    and returns distances for each kernel radius across all wells.

    Parameters
    ----------
    dir_results : str, optional
        Directory containing sensitivity analysis results. Default is the standard path.
    output_dir : str, optional
        Directory to save per-well distance results. If None, results are not saved.
        If provided, each well's distances will be saved as a numpy archive.

    Returns
    -------
    Dict[str, Dict[int, np.ndarray]]
        Nested dictionary where:
        - Outer keys are well identifiers in format "well_id_plate_id"
        - Inner keys are kernel radii
        - Values are numpy arrays of distances for each nucleus

    Examples
    --------
    >>> all_distances = calculate_kernel_distances_all_wells()
    >>> well_key = list(all_distances.keys())[0]
    >>> print(f"Well {well_key}, Radius 5: mean distance = {all_distances[well_key][5].mean():.4f}")

    >>> # Save results to disk
    >>> all_distances = calculate_kernel_distances_all_wells(output_dir="./distance_results")
    """
    dir_results = Path(dir_results)

    if not dir_results.exists():
        raise FileNotFoundError(f"Results directory not found: {dir_results}")

    # Find all sensitivity analysis files
    result_files = sorted(dir_results.glob("*_sensitivity_analysis.h5ad"))

    if not result_files:
        raise FileNotFoundError(f"No sensitivity analysis files found in {dir_results}")
    if n is not None:
        result_files = result_files[:n]

    # Create output directory if specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Process all wells
    all_distances = {}

    for file_path in tqdm(result_files, desc="Calculating distances for all wells"):
        # Extract well_id and plate_id from filename
        file_stem = file_path.stem  # Remove .h5ad extension
        well_id, plate_id = file_stem.rsplit("_sensitivity_analysis", 1)[0].rsplit("_", 1)

        # Calculate distances for this well
        try:
            distances = calculate_kernel_distances_for_well(well_id, plate_id, dir_results)
            well_key = f"{well_id}_{plate_id}"
            all_distances[well_key] = distances

            # Save to disk if output directory specified
            if output_dir is not None:
                output_file = output_dir / f"{well_key}_distances.npz"
                # Convert to regular dict for saving
                distances_dict = {str(radius): distances[radius] for radius in distances.keys()}
                np.savez_compressed(output_file, **distances_dict)

        except Exception as e:
            print(f"Warning: Failed to process {file_path.name}: {str(e)}")
            continue

    print(f"Successfully processed {len(all_distances)} wells")

    return all_distances


def load_kernel_distances_from_files(
    distances_dir: str = "./sensitivity_analysis_distances",
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load all pre-calculated kernel distances from saved .npz files.

    This function loads distance data that was previously saved by
    calculate_kernel_distances_all_wells() with the output_dir parameter.
    Use this to skip recalculation if distances have already been computed.

    Parameters
    ----------
    distances_dir : str, optional
        Directory containing the saved distance files. Expected to contain
        files named "{well_id}_{plate_id}_distances.npz". Default is
        "./sensitivity_analysis_distances".

    Returns
    -------
    Dict[str, Dict[int, np.ndarray]]
        Nested dictionary where:
        - Outer keys are well identifiers ("well_id_plate_id")
        - Inner keys are kernel radii (as integers)
        - Values are numpy arrays of per-nucleus distances

    Raises
    ------
    FileNotFoundError
        If the distances directory does not exist or contains no .npz files.

    Examples
    --------
    >>> all_distances = load_kernel_distances_from_files(
    ...     distances_dir="./sensitivity_analysis_distances"
    ... )
    >>> print(f"Loaded distances for {len(all_distances)} wells")

    >>> fig = plot_kernel_distances_violin(all_distances)
    >>> plt.show()
    """
    distances_dir = Path(distances_dir)

    if not distances_dir.exists():
        raise FileNotFoundError(f"Distances directory not found: {distances_dir}")

    all_distances = {}
    distance_files = list(distances_dir.glob("*_distances.npz"))

    if not distance_files:
        raise FileNotFoundError(
            f"No distance files found in {distances_dir}. Expected files matching pattern '*_distances.npz'"
        )

    print(f"Loading distances from {len(distance_files)} files...")

    for npz_file in tqdm(sorted(distance_files), desc="Loading distances"):
        well_key = npz_file.stem.replace("_distances", "")
        loaded_data = np.load(npz_file)
        distances_dict = {}
        for key in loaded_data.files:
            radius = int(key)
            distances_dict[radius] = loaded_data[key]
        all_distances[well_key] = distances_dict

    print(f"Successfully loaded distances for {len(all_distances)} wells")
    return all_distances


def plot_kernel_distances_violin(
    all_distances: Dict[str, Dict[int, np.ndarray]],
    figsize: tuple = (14, 15),
    palette: str = "Set2",
    title: str = "Kernel Size Sensitivity Analysis: Distance Distributions",
    output_path: str = None,
    dpi: int = 300,
    quantile_clip: tuple = (0, 97.5),
) -> plt.Figure:
    """
    Plot three-panel violin plots showing distance distributions across kernel sizes.

    Creates a figure with three stacked rows:
    1. Top: Distribution of individual per-nucleus distances (with quantile clipping)
    2. Middle: Distribution of median distances per well across samples
    3. Bottom: Distribution of mean distances per well across samples

    Parameters
    ----------
    all_distances : Dict[str, Dict[int, np.ndarray]]
        Nested dictionary from calculate_kernel_distances_all_wells() where:
        - Outer keys are well identifiers ("well_id_plate_id")
        - Inner keys are kernel radii
        - Values are arrays of per-nucleus distances
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (14, 15).
    palette : str, optional
        Seaborn color palette. Default is "Set2".
    title : str, optional
        Figure title. Default describes the analysis.
    output_path : str, optional
        Path to save the figure. If None, figure is not saved. Should end with .png, .pdf, etc.
    dpi : int, optional
        Resolution for saved figure. Default is 300 dpi.
    quantile_clip : tuple, optional
        Quantile range for clipping outliers in per-nucleus distances panel (lower, upper).
        Default is (0, 97.5) to clip 0th and 97.5th percentiles.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing both violin plots.

    Examples
    --------
    >>> all_distances = calculate_kernel_distances_all_wells()
    >>> fig = plot_kernel_distances_violin(all_distances)
    >>> plt.show()

    >>> fig = plot_kernel_distances_violin(
    ...     all_distances,
    ...     output_path="./sensitivity_analysis.png",
    ...     quantile_clip=(5, 95)
    ... )
    """
    # Get all radii (sorted)
    all_radii = set()
    for distances_dict in all_distances.values():
        all_radii.update(distances_dict.keys())
    all_radii = sorted(all_radii)

    # Panel 1: Per-nucleus distances with quantile clipping
    plot_data_per_nucleus = []

    for well_key, distances_dict in all_distances.items():
        for radius in all_radii:
            if radius in distances_dict:
                distances_array = distances_dict[radius]
                lower, upper = quantile_clip
                lower_clip = np.percentile(distances_array, lower)
                upper_clip = np.percentile(distances_array, upper)
                clipped_distances = np.clip(distances_array, lower_clip, upper_clip)

                for distance in clipped_distances:
                    plot_data_per_nucleus.append(
                        {
                            "Kernel Radius": radius,
                            "Distance": distance,
                            "Well": well_key,
                        }
                    )

    # Panel 2: Median distances per well
    plot_data_median = []

    for well_key, distances_dict in all_distances.items():
        for radius in all_radii:
            if radius in distances_dict:
                distances_array = distances_dict[radius]
                median_distance = np.median(distances_array)
                plot_data_median.append(
                    {
                        "Kernel Radius": radius,
                        "Median Distance": median_distance,
                        "Well": well_key,
                    }
                )

    # Panel 3: Mean distances per well
    plot_data_mean = []

    for well_key, distances_dict in all_distances.items():
        for radius in all_radii:
            if radius in distances_dict:
                distances_array = distances_dict[radius]
                mean_distance = np.mean(distances_array)
                plot_data_mean.append(
                    {
                        "Kernel Radius": radius,
                        "Mean Distance": mean_distance,
                        "Well": well_key,
                    }
                )

    # Create DataFrames
    df_per_nucleus = pd.DataFrame(plot_data_per_nucleus)
    df_median = pd.DataFrame(plot_data_median)
    df_mean = pd.DataFrame(plot_data_mean)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

    # Top panel: Per-nucleus distances
    sns.violinplot(
        data=df_per_nucleus,
        x="Kernel Radius",
        y="Distance",
        palette=palette,
        ax=ax1,
    )
    ax1.set_title(
        f"Per-Nucleus Distances (Clipped {quantile_clip[0]}-{quantile_clip[1]}%)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xlabel("Kernel Radius", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Euclidean Distance", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_xticklabels([str(r) for r in all_radii], rotation=0)

    # Middle panel: Median distances per well
    sns.violinplot(
        data=df_median,
        x="Kernel Radius",
        y="Median Distance",
        palette=palette,
        ax=ax2,
    )
    ax2.set_title(
        "Median Distance per Well",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xlabel("Kernel Radius", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Median Euclidean Distance", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.set_xticklabels([str(r) for r in all_radii], rotation=0)

    # Bottom panel: Mean distances per well
    sns.violinplot(
        data=df_mean,
        x="Kernel Radius",
        y="Mean Distance",
        palette=palette,
        ax=ax3,
    )
    ax3.set_title(
        "Mean Distance per Well",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax3.set_xlabel("Kernel Radius", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Mean Euclidean Distance", fontsize=11, fontweight="bold")
    ax3.grid(axis="y", alpha=0.3, linestyle="--")
    ax3.set_xticklabels([str(r) for r in all_radii], rotation=0)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save figure if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    return fig


def plot_kernel_distances_violin_by_well(
    all_distances: Dict[str, Dict[int, np.ndarray]],
    figsize: tuple = None,
    palette: str = "Set2",
    output_path: str = None,
    dpi: int = 300,
) -> Dict[str, plt.Figure]:
    """
    Create separate violin plots for each well showing kernel size sensitivity.

    Creates one figure per well, useful for comparing how different wells respond
    to kernel size changes.

    Parameters
    ----------
    all_distances : Dict[str, Dict[int, np.ndarray]]
        Nested dictionary from calculate_kernel_distances_all_wells().
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (14, 5).
    palette : str, optional
        Seaborn color palette. Default is "Set2".
    output_path : str, optional
        Directory to save figures. If None, figures are not saved.
        Figures will be saved as "well_id_plate_id_distances.png".
    dpi : int, optional
        Resolution for saved figures. Default is 300 dpi.

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary mapping well identifiers to their matplotlib Figure objects.

    Examples
    --------
    >>> all_distances = calculate_kernel_distances_all_wells()
    >>> figs = plot_kernel_distances_violin_by_well(
    ...     all_distances,
    ...     output_path="./well_plots"
    ... )
    >>> # Figs contains one figure per well
    """
    if figsize is None:
        figsize = (14, 5)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

    figures = {}

    for well_key, distances_dict in all_distances.items():
        # Get sorted radii for this well
        radii = sorted(distances_dict.keys())

        # Prepare data
        plot_data = []
        for radius in radii:
            distances_array = distances_dict[radius]
            for distance in distances_array:
                plot_data.append({"Kernel Radius": radius, "Distance": distance})

        df = pd.DataFrame(plot_data)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        sns.violinplot(
            data=df,
            x="Kernel Radius",
            y="Distance",
            palette=palette,
            ax=ax,
        )

        ax.set_title(
            f"Kernel Sensitivity: {well_key}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Kernel Radius", fontsize=12, fontweight="bold")
        ax.set_ylabel("Euclidean Distance from Baseline", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xticklabels([str(r) for r in radii], rotation=0)

        plt.tight_layout()

        figures[well_key] = fig

        # Save if output path provided
        if output_path is not None:
            fig_path = output_path / f"{well_key}_distances.png"
            fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")

    print(f"Created {len(figures)} well-specific figures")

    return figures


def plot_kernel_distances_aggregated_stats(
    all_distances: Dict[str, Dict[int, np.ndarray]],
    figsize: tuple = (14, 7),
    palette: str = "husl",
    output_path: str = None,
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot aggregated statistics (mean, median, std) across all wells for each kernel size.

    Creates a comprehensive overview with multiple subplots showing how the distance
    distribution changes across kernel radii, with error bars and trend lines.

    Parameters
    ----------
    all_distances : Dict[str, Dict[int, np.ndarray]]
        Nested dictionary from calculate_kernel_distances_all_wells().
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (14, 7).
    palette : str, optional
        Seaborn color palette. Default is "husl".
    output_path : str, optional
        Path to save the figure. If None, figure is not saved.
    dpi : int, optional
        Resolution for saved figure. Default is 300 dpi.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing multiple subplots.

    Examples
    --------
    >>> all_distances = calculate_kernel_distances_all_wells()
    >>> fig = plot_kernel_distances_aggregated_stats(all_distances)
    >>> plt.show()

    >>> # Save to file
    >>> fig = plot_kernel_distances_aggregated_stats(
    ...     all_distances,
    ...     output_path="./sensitivity_aggregated_stats.png"
    ... )
    """
    # Get all radii (sorted)
    all_radii = set()
    for distances_dict in all_distances.values():
        all_radii.update(distances_dict.keys())
    all_radii = sorted(all_radii)

    # Compute statistics for each radius
    stats = {}
    for radius in all_radii:
        all_dists = []
        for distances_dict in all_distances.values():
            if radius in distances_dict:
                all_dists.extend(distances_dict[radius])
        all_dists = np.array(all_dists)

        # Clip data at 1 and 99 percentiles to remove extreme outliers
        p1 = np.percentile(all_dists, 1)
        p99 = np.percentile(all_dists, 99)
        all_dists_clipped = np.clip(all_dists, p1, p99)

        stats[radius] = {
            "mean": np.mean(all_dists_clipped),
            "median": np.median(all_dists_clipped),
            "std": np.std(all_dists_clipped),
            "q25": np.percentile(all_dists_clipped, 25),
            "q75": np.percentile(all_dists_clipped, 75),
            "min": np.percentile(all_dists_clipped, 1),
            "max": np.percentile(all_dists_clipped, 99),
            "n": len(all_dists),
        }

    # Extract data for plotting
    radii_list = sorted(stats.keys())
    means = [stats[r]["mean"] for r in radii_list]
    medians = [stats[r]["median"] for r in radii_list]
    stds = [stats[r]["std"] for r in radii_list]
    q25s = [stats[r]["q25"] for r in radii_list]
    q75s = [stats[r]["q75"] for r in radii_list]
    mins = [stats[r]["min"] for r in radii_list]
    maxs = [stats[r]["max"] for r in radii_list]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Color for the lines
    color_mean = sns.color_palette(palette, 4)[0]
    color_median = sns.color_palette(palette, 4)[1]
    color_range = sns.color_palette(palette, 4)[2]
    color_quartile = sns.color_palette(palette, 4)[3]

    # Subplot 1: Mean with standard deviation
    ax = axes[0, 0]
    ax.plot(
        radii_list,
        means,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Mean",
        color=color_mean,
    )
    ax.fill_between(
        radii_list,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.3,
        color=color_mean,
        label="±1 SD",
    )
    ax.set_xlabel("Kernel Radius", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=11, fontweight="bold")
    ax.set_title("Mean Distance ± Std Dev", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()

    # Subplot 2: Mean vs Median
    ax = axes[0, 1]
    ax.plot(
        radii_list,
        means,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Mean",
        color=color_mean,
    )
    ax.plot(
        radii_list,
        medians,
        marker="s",
        linewidth=2.5,
        markersize=8,
        label="Median",
        color=color_median,
    )
    ax.set_xlabel("Kernel Radius", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=11, fontweight="bold")
    ax.set_title("Mean vs Median", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()

    # Subplot 3: Min-Max range with quartiles
    ax = axes[1, 0]
    ax.fill_between(
        radii_list,
        mins,
        maxs,
        alpha=0.2,
        color=color_range,
        label="Min-Max Range",
    )
    ax.fill_between(
        radii_list,
        q25s,
        q75s,
        alpha=0.4,
        color=color_quartile,
        label="IQR (Q1-Q3)",
    )
    ax.plot(
        radii_list,
        medians,
        marker="D",
        linewidth=2,
        markersize=7,
        color="black",
        label="Median",
        zorder=10,
    )
    ax.set_xlabel("Kernel Radius", fontsize=11, fontweight="bold")
    ax.set_ylabel("Distance", fontsize=11, fontweight="bold")
    ax.set_title("Distribution Range (Min-Max, IQR)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()

    # Subplot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis("off")

    # Create summary table
    summary_data = []
    summary_data.append(["Radius", "Mean", "Median", "Std", "N Nuclei"])
    for radius in radii_list:
        summary_data.append(
            [
                str(radius),
                f"{stats[radius]['mean']:.3f}",
                f"{stats[radius]['median']:.3f}",
                f"{stats[radius]['std']:.3f}",
                f"{stats[radius]['n']:,}",
            ]
        )

    table = ax.table(
        cellText=summary_data,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(summary_data)):
        for j in range(len(summary_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")
            else:
                table[(i, j)].set_facecolor("white")

    ax.set_title("Summary Statistics", fontsize=12, fontweight="bold", pad=20)

    # Overall title
    fig.suptitle(
        "Kernel Size Sensitivity Analysis: Aggregated Statistics Across All Wells",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save figure if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    return fig


def main(
    dir_results: str = "data/pilotscreen/anndata/sensitivity_analysis_results",
    output_dir: str = "./sensitivity_analysis_plots",
    save_distances: bool = True,
    distances_output_dir: str = "./sensitivity_analysis_distances",
):
    """
    Run the complete sensitivity analysis pipeline: calculate distances and generate plots.

    This function orchestrates the full workflow:
    1. Calculate distances for all wells
    2. Generate aggregated violin plot
    3. Generate per-well violin plots
    4. Generate aggregated statistics plots
    5. Optionally save distance data

    Parameters
    ----------
    dir_results : str, optional
        Directory containing sensitivity analysis results. Default is the standard path.
    output_dir : str, optional
        Directory to save all plot figures. Default is "./sensitivity_analysis_plots".
    save_distances : bool, optional
        Whether to save distance arrays to disk. Default is True.
    distances_output_dir : str, optional
        Directory to save distance data. Default is "./sensitivity_analysis_distances".

    Examples
    --------
    >>> main()
    >>> # or with custom output paths
    >>> main(
    ...     output_dir="./results/plots",
    ...     distances_output_dir="./results/distances"
    ... )
    """
    plt.rcParams["pdf.fonttype"] = 42

    print("=" * 60)
    print("Sensitivity Analysis: Distance Calculation and Visualization")
    print("=" * 60)

    # Step 1: Calculate distances for all wells
    print("\n[1/4] Calculating kernel distances for all wells...")
    distances_out_dir = distances_output_dir if save_distances else None
    all_distances = calculate_kernel_distances_all_wells(
        dir_results=dir_results,
        output_dir=distances_out_dir,
    )
    print(f"✓ Processed {len(all_distances)} wells")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Aggregated violin plot
    print("\n[2/4] Creating aggregated violin plot...")
    fig_violin = plot_kernel_distances_violin(
        all_distances,
        output_path=output_dir / "01_aggregated_violin.pdf",
    )
    print("✓ Aggregated violin plot created")

    # Step 3: Per-well violin plots
    print("\n[3/4] Creating per-well violin plots...")
    well_plots_dir = output_dir / "per_well_plots"
    figs_by_well = plot_kernel_distances_violin_by_well(all_distances, output_path=well_plots_dir)
    print(f"✓ Created {len(figs_by_well)} per-well plots")

    # Step 4: Aggregated statistics plots
    print("\n[4/4] Creating aggregated statistics plot...")
    fig_stats = plot_kernel_distances_aggregated_stats(
        all_distances,
        output_path=output_dir / "02_aggregated_statistics.pdf",
    )
    print("✓ Aggregated statistics plot created")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"  - Aggregated violin plot: 01_aggregated_violin.png")
    print(f"  - Aggregated statistics: 02_aggregated_statistics.png")
    print(f"  - Per-well plots: per_well_plots/ ({len(figs_by_well)} files)")
    if save_distances:
        print(f"\nDistance data saved to: {distances_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

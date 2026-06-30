"""
Module for comparing and analyzing segmentation label images.

This module provides functions to compare cellpose automated segmentation output
with manually corrected segmentation (ground truth) for 3D images. It computes
evaluation metrics including F1 scores, precision, recall, and IoU at both
sample-level and per-slice level.
"""

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = intersection + np.logical_xor(pred_mask, true_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_dice(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, true_mask).sum()
    return float(2 * intersection / (pred_mask.sum() + true_mask.sum() + 1e-8))


def compute_precision_recall(
    pred_mask: np.ndarray, true_mask: np.ndarray
) -> Tuple[float, float]:
    tp = np.logical_and(pred_mask, true_mask).sum()
    fp = np.logical_and(pred_mask, ~true_mask.astype(bool)).sum()
    fn = np.logical_and(~pred_mask.astype(bool), true_mask).sum()
    precision = float(tp / (tp + fp + 1e-8))
    recall = float(tp / (tp + fn + 1e-8))
    return precision, recall


def match_labels_by_iou(
    pred_labels: np.ndarray, true_labels: np.ndarray, iou_threshold: float = 0.1
) -> Dict[int, int]:
    """
    Match predicted labels to ground truth labels using IoU overlap.

    Vectorised: intersection counts come from np.unique on encoded voxel pairs,
    label sizes from np.bincount — no per-pair boolean mask creation.
    Uses the Hungarian algorithm for optimal matching.

    :param pred_labels: Predicted 2D label image
    :param true_labels: Ground truth 2D label image
    :param iou_threshold: Minimum IoU to consider labels as matched
    :return: Dictionary mapping predicted label IDs to true label IDs
    """
    pred_ids = np.unique(pred_labels[pred_labels > 0])
    true_ids = np.unique(true_labels[true_labels > 0])

    if len(pred_ids) == 0 or len(true_ids) == 0:
        return {}

    overlap_mask = (pred_labels > 0) & (true_labels > 0)
    if not overlap_mask.any():
        return {}

    # Encode overlapping voxel pairs as a single integer, count intersections
    max_true = int(true_ids.max()) + 1
    pair_codes = pred_labels[overlap_mask].astype(np.int64) * max_true + true_labels[
        overlap_mask
    ].astype(np.int64)
    codes, intersection_counts = np.unique(pair_codes, return_counts=True)
    pairs_pred = (codes // max_true).astype(int)
    pairs_true = (codes % max_true).astype(int)

    # Label sizes via bincount (one pass each)
    pred_sizes = np.bincount(pred_labels.ravel(), minlength=int(pred_ids.max()) + 1)
    true_sizes = np.bincount(true_labels.ravel(), minlength=max_true)

    pred_id_to_idx = {int(pid): i for i, pid in enumerate(pred_ids)}
    true_id_to_idx = {int(tid): j for j, tid in enumerate(true_ids)}

    iou_matrix = np.zeros((len(pred_ids), len(true_ids)))
    for pred_id, true_id, inter in zip(pairs_pred, pairs_true, intersection_counts):
        i = pred_id_to_idx.get(pred_id)
        j = true_id_to_idx.get(true_id)
        if i is None or j is None:
            continue
        union = pred_sizes[pred_id] + true_sizes[true_id] - inter
        iou_matrix[i, j] = inter / union if union > 0 else 0.0

    pred_indices, true_indices = linear_sum_assignment(-iou_matrix)

    matches = {}
    for pred_idx, true_idx in zip(pred_indices, true_indices):
        if iou_matrix[pred_idx, true_idx] >= iou_threshold:
            matches[int(pred_ids[pred_idx])] = int(true_ids[true_idx])

    return matches


def compute_instance_metrics(
    pred_count: int, true_count: int, matched_count: int
) -> Dict[str, float]:
    """
    Compute instance-level detection precision, recall, and F1.

    These are the meaningful metrics for nucleus detection: they count cells,
    not pixels. Precision = matched / pred (of detected cells, how many are real);
    Recall = matched / true (of real cells, how many were detected).

    :param pred_count: Number of predicted cell instances
    :param true_count: Number of ground truth cell instances
    :param matched_count: Number of matched pairs (TP)
    :return: Dictionary with instance_precision, instance_recall, instance_f1
    """
    inst_precision = matched_count / (pred_count + 1e-8) if pred_count > 0 else 0.0
    inst_recall = matched_count / (true_count + 1e-8) if true_count > 0 else 0.0
    inst_f1 = 2 * (inst_precision * inst_recall) / (inst_precision + inst_recall + 1e-8)
    return {
        "instance_precision": float(inst_precision),
        "instance_recall": float(inst_recall),
        "instance_f1": float(inst_f1),
    }


def compare_label_images(
    pred_labels: np.ndarray, true_labels: np.ndarray, iou_threshold: float = 0.1
) -> Dict[str, Union[float, int]]:
    """
    Compare predicted and ground truth 2D label images and compute evaluation metrics.

    Reports instance-level detection metrics (precision/recall/F1 based on matched
    cell counts) as the primary output, alongside pixel-overlap metrics for reference.

    :param pred_labels: Predicted 2D label image
    :param true_labels: Ground truth 2D label image
    :param iou_threshold: Minimum IoU threshold for label matching
    :return: Dictionary containing computed metrics
    """
    matches = match_labels_by_iou(pred_labels, true_labels, iou_threshold)

    pred_count = len(np.unique(pred_labels[pred_labels > 0]))
    true_count = len(np.unique(true_labels[true_labels > 0]))
    matched_count = len(matches)

    instance_metrics = compute_instance_metrics(pred_count, true_count, matched_count)

    # Pixel-overlap metrics — note these are mask-overlap-given-match and are
    # circular when GT masks are derived from the same model output
    pred_binary = pred_labels > 0
    true_binary = true_labels > 0
    pixel_dice = compute_dice(pred_binary, true_binary)
    pixel_iou = compute_iou(pred_binary, true_binary)
    pixel_precision, pixel_recall = compute_precision_recall(pred_binary, true_binary)

    return {
        # Instance detection metrics (headline numbers)
        "instance_f1": instance_metrics["instance_f1"],
        "instance_precision": instance_metrics["instance_precision"],
        "instance_recall": instance_metrics["instance_recall"],
        "pred_cell_count": int(pred_count),
        "true_cell_count": int(true_count),
        "matched_cells": int(matched_count),
        "false_positives": int(pred_count - matched_count),
        "false_negatives": int(true_count - matched_count),
        # Pixel-overlap metrics (supplementary; circular if GT derived from same model)
        "pixel_dice": float(pixel_dice),
        "pixel_iou": float(pixel_iou),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
    }


def compare_label_images_per_slice(
    pred_labels: np.ndarray, true_labels: np.ndarray, iou_threshold: float = 0.1
) -> pd.DataFrame:
    """
    Compare predicted and ground truth 3D label images slice-by-slice.

    :param pred_labels: Predicted 3D label image (z, y, x)
    :param true_labels: Ground truth 3D label image (z, y, x)
    :param iou_threshold: Minimum IoU threshold for label matching
    :return: DataFrame with per-slice metrics (one row per z-slice)
    """
    if pred_labels.ndim != 3 or true_labels.ndim != 3:
        raise ValueError("Input arrays must be 3D (z, y, x)")
    if pred_labels.shape != true_labels.shape:
        raise ValueError("Predicted and true labels must have the same shape")

    results = []
    for z in tqdm(range(pred_labels.shape[0]), desc="per-slice matching"):
        metrics = compare_label_images(pred_labels[z], true_labels[z], iou_threshold)
        metrics["z_slice"] = z
        results.append(metrics)

    df = pd.DataFrame(results)
    cols = ["z_slice"] + [c for c in df.columns if c != "z_slice"]
    return df[cols]


def get_comparison_summary(
    pred_labels: np.ndarray, true_labels: np.ndarray, iou_threshold: float = 0.1
) -> Tuple[Dict, pd.DataFrame]:
    """
    Compare 3D label images and return both sample-level and per-slice metrics.

    Sample-level metrics use binary pixel comparisons on the full volume (no 3D
    label matching). Per-slice cell counts use per-slice IoU matching.

    :param pred_labels: Predicted 3D label image (z, y, x)
    :param true_labels: Ground truth 3D label image (z, y, x)
    :param iou_threshold: Minimum IoU threshold for label matching
    :return: Tuple of (sample_metrics_dict, per_slice_metrics_dataframe)
    """
    if pred_labels.ndim != 3 or true_labels.ndim != 3:
        raise ValueError("Input arrays must be 3D (z, y, x)")
    if pred_labels.shape != true_labels.shape:
        raise ValueError("Predicted and true labels must have the same shape")

    print(f"[get_comparison_summary] volume shape: {pred_labels.shape}", flush=True)

    print("[get_comparison_summary] step 1/2: 3D volume-level metrics...", flush=True)
    pred_binary = pred_labels > 0
    true_binary = true_labels > 0
    dice = compute_dice(pred_binary, true_binary)
    iou_score = compute_iou(pred_binary, true_binary)
    precision, recall = compute_precision_recall(pred_binary, true_binary)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    pred_count_3d = int(len(np.unique(pred_labels[pred_labels > 0])))
    true_count_3d = int(len(np.unique(true_labels[true_labels > 0])))
    print(f"  labels — pred: {pred_count_3d}, true: {true_count_3d}", flush=True)

    print("  matching 3D labels (sparse IoU)...", flush=True)
    matches_3d = match_labels_by_iou(pred_labels, true_labels, iou_threshold)
    matched_count_3d = len(matches_3d)
    print(f"  matched: {matched_count_3d}", flush=True)

    # 3D instance detection metrics
    inst_3d = compute_instance_metrics(pred_count_3d, true_count_3d, matched_count_3d)

    # 3D pixel metrics (supplementary)
    pred_binary = pred_labels > 0
    true_binary = true_labels > 0
    pixel_dice = compute_dice(pred_binary, true_binary)
    pixel_iou = compute_iou(pred_binary, true_binary)
    pixel_precision, pixel_recall = compute_precision_recall(pred_binary, true_binary)

    sample_metrics: Dict[str, Union[float, int]] = {
        # Instance detection (headline)
        "instance_f1": inst_3d["instance_f1"],
        "instance_precision": inst_3d["instance_precision"],
        "instance_recall": inst_3d["instance_recall"],
        "pred_cell_count": pred_count_3d,
        "true_cell_count": true_count_3d,
        "matched_cells": matched_count_3d,
        "false_positives": pred_count_3d - matched_count_3d,
        "false_negatives": true_count_3d - matched_count_3d,
        # Pixel overlap (supplementary)
        "pixel_dice": float(pixel_dice),
        "pixel_iou": float(pixel_iou),
        "pixel_precision": float(pixel_precision),
        "pixel_recall": float(pixel_recall),
    }
    print("[get_comparison_summary] step 1/2 done.", flush=True)

    print(
        f"[get_comparison_summary] step 2/2: per-slice matching ({pred_labels.shape[0]} slices)...",
        flush=True,
    )
    per_slice_df = compare_label_images_per_slice(
        pred_labels, true_labels, iou_threshold
    )
    print("[get_comparison_summary] step 2/2 done.", flush=True)

    sample_metrics["n_z_slices"] = pred_labels.shape[0]
    # Per-slice instance F1 mean ± std — the number for the paper
    sample_metrics["mean_instance_f1_per_slice"] = float(
        per_slice_df["instance_f1"].mean()
    )
    sample_metrics["std_instance_f1_per_slice"] = float(
        per_slice_df["instance_f1"].std()
    )
    sample_metrics["mean_instance_precision_per_slice"] = float(
        per_slice_df["instance_precision"].mean()
    )
    sample_metrics["mean_instance_recall_per_slice"] = float(
        per_slice_df["instance_recall"].mean()
    )
    sample_metrics["mean_pixel_dice_per_slice"] = float(
        per_slice_df["pixel_dice"].mean()
    )

    return sample_metrics, per_slice_df


def format_metrics_table(metrics_dict: Dict, decimal_places: int = 4) -> Dict:
    return {
        k: round(v, decimal_places) if isinstance(v, float) else v
        for k, v in metrics_dict.items()
    }


def print_comparison_results(sample_metrics: Dict, per_slice_df: pd.DataFrame = None):
    print("\n" + "=" * 60)
    print("SEGMENTATION COMPARISON RESULTS")
    print("=" * 60)

    instance_keys = {
        "instance_f1",
        "instance_precision",
        "instance_recall",
        "pred_cell_count",
        "true_cell_count",
        "matched_cells",
        "false_positives",
        "false_negatives",
        "mean_instance_f1_per_slice",
        "std_instance_f1_per_slice",
        "mean_instance_precision_per_slice",
        "mean_instance_recall_per_slice",
        "n_z_slices",
    }
    pixel_keys = {
        "pixel_dice",
        "pixel_iou",
        "pixel_precision",
        "pixel_recall",
        "mean_pixel_dice_per_slice",
    }

    print("\nInstance Detection Metrics (headline):")
    print("-" * 60)
    for key, value in sample_metrics.items():
        if key in instance_keys:
            if isinstance(value, float):
                print(f"  {key:.<44} {value:.4f}")
            else:
                print(f"  {key:.<44} {value}")

    print("\nPixel Overlap Metrics (supplementary; circular if GT from same model):")
    print("-" * 60)
    for key, value in sample_metrics.items():
        if key in pixel_keys:
            print(f"  {key:.<44} {value:.4f}")

    if per_slice_df is not None and len(per_slice_df) > 0:
        instance_cols = [
            "z_slice",
            "instance_f1",
            "instance_precision",
            "instance_recall",
            "pred_cell_count",
            "true_cell_count",
            "matched_cells",
            "false_positives",
            "false_negatives",
        ]
        present = [c for c in instance_cols if c in per_slice_df.columns]

        print("\n\nPer-slice Instance Metrics — summary:")
        print("-" * 60)
        print(per_slice_df[present].describe().to_string())

        print("\n\nPer-slice Instance Metrics — first 10 slices:")
        print("-" * 60)
        print(per_slice_df[present].head(10).to_string())


if __name__ == "__main__":
    np.random.seed(42)

    z, y, x = 10, 100, 100
    true_labels = np.zeros((z, y, x), dtype=np.uint32)
    cell_id = 1
    for zz in range(z):
        for _ in range(5):
            cy, cx = np.random.randint(20, y - 20), np.random.randint(20, x - 20)
            radius = np.random.randint(5, 15)
            yy, xx = np.ogrid[:y, :x]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2
            true_labels[zz][mask] = cell_id
            cell_id += 1

    pred_labels = true_labels.copy()
    noise_mask = np.random.rand(z, y, x) < 0.05
    pred_labels[noise_mask] = 0

    sample_metrics, per_slice_df = get_comparison_summary(pred_labels, true_labels)
    print_comparison_results(sample_metrics, per_slice_df)

    from cellpose.utils import stitch3D
    from skimage import io
    from skimage.measure import label as relabel

    labels_init = io.imread("test_label_og.tif")
    labels_corr = io.imread("test_to_z33.tif")

    # Binarize, relabel each slice independently, then stitch across z
    print("Binarizing and relabelling labels_corr per slice...", flush=True)
    binary_corr = labels_corr > 0
    per_slice_labels = np.stack(
        [
            relabel(binary_corr[z]).astype(np.uint32)
            for z in range(binary_corr.shape[0])
        ],
        axis=0,
    )
    print("Stitching across z with cellpose stitch3D...", flush=True)
    labels_corr_stitched = stitch3D(per_slice_labels, stitch_threshold=0.25)
    print(
        f"Stitching done. Unique labels: {len(np.unique(labels_corr_stitched)) - 1}",
        flush=True,
    )

    sample_metrics_corr, per_slice_df_corr = get_comparison_summary(
        labels_init, labels_corr_stitched
    )
    print_comparison_results(sample_metrics_corr, per_slice_df_corr)

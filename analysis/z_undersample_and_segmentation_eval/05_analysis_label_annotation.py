"""
Compare two 3D instance-label volumes (e.g. Cellpose prediction vs. semi-manually
corrected ground truth).

Reports:
  * instance detection:  TP / FP / FN, precision, recall, F1, AP (=TP/(TP+FP+FN))
                         at one or more IoU thresholds
  * pixel / semantic:    foreground IoU, Dice, mean IoU over matched instances
  * global (pooled):     micro- and macro-averages + across-sample SD

Designed for big volumes (e.g. 50 x 3840 x 3840): the label contingency table is
accumulated slice-by-slice as a sparse matrix, and background-background pixels
are never materialised.

Convention: `labels_gt` (the corrected mask) is truth, `labels_pred` is the
prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


# --------------------------------------------------------------------------- #
# core: sparse contingency table
# --------------------------------------------------------------------------- #
def _slice_contingency(gt_2d, pred_2d, n_gt, n_pred):
    """Sparse (n_gt+1, n_pred+1) pixel-overlap counts for one slice.

    Row/col 0 = background. Pixels that are background in *both* volumes are
    dropped, which is what keeps this cheap on 3840x3840 slices.
    """
    x = gt_2d.ravel()
    y = pred_2d.ravel()

    keep = (x > 0) | (y > 0)
    x = x[keep].astype(np.int64, copy=False)
    y = y[keep].astype(np.int64, copy=False)

    data = np.ones(x.size, dtype=np.int64)
    m = coo_matrix((data, (x, y)), shape=(n_gt + 1, n_pred + 1))
    return m.tocsr()  # tocsr() sums duplicate entries


def _contingency(labels_gt, labels_pred, per_slice=True):
    """Accumulate the full-volume contingency matrix, yielding per-slice ones."""
    n_gt = int(labels_gt.max())
    n_pred = int(labels_pred.max())

    total = csr_matrix((n_gt + 1, n_pred + 1), dtype=np.int64)
    per_slice_mats = []

    for z in range(labels_gt.shape[0]):
        m = _slice_contingency(labels_gt[z], labels_pred[z], n_gt, n_pred)
        total = total + m
        if per_slice:
            per_slice_mats.append(m)

    return total, per_slice_mats, n_gt, n_pred


def _iou_pairs(cont):
    """Return (gt_idx, pred_idx, iou) for every pair with non-zero overlap.

    Indices are 1-based label ids. Areas are taken from the full contingency
    matrix (including the background row/col) so they are exact.
    """
    area_gt = np.asarray(cont.sum(axis=1)).ravel()   # includes overlap with bg
    area_pred = np.asarray(cont.sum(axis=0)).ravel()

    ov = cont[1:, 1:].tocoo()
    if ov.nnz == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([], dtype=np.float64), area_gt, area_pred

    gt_ids = ov.row + 1
    pred_ids = ov.col + 1
    inter = ov.data.astype(np.float64)
    union = area_gt[gt_ids] + area_pred[pred_ids] - inter
    iou = inter / union
    return gt_ids, pred_ids, iou, area_gt, area_pred


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def _match(gt_ids, pred_ids, iou, tau, max_hungarian=8000):
    """TP count + IoUs of matched pairs at threshold `tau`.

    tau >= 0.5  -> matching is provably unique, just threshold (O(nnz)).
    tau <  0.5  -> optimal assignment required (Hungarian on a dense submatrix).
    """
    if iou.size == 0:
        return 0, np.array([], dtype=np.float64)

    if tau >= 0.5:
        keep = iou >= tau
        return int(keep.sum()), iou[keep]

    # --- tau < 0.5: optimal bipartite assignment on candidate pairs only ---
    from scipy.optimize import linear_sum_assignment

    keep = iou >= tau
    g, p, v = gt_ids[keep], pred_ids[keep], iou[keep]
    if v.size == 0:
        return 0, v

    urow, ri = np.unique(g, return_inverse=True)
    ucol, ci = np.unique(p, return_inverse=True)
    if urow.size > max_hungarian or ucol.size > max_hungarian:
        raise MemoryError(
            f"Hungarian matrix would be {urow.size}x{ucol.size}; "
            f"use tau >= 0.5 or raise max_hungarian."
        )

    cost = np.zeros((urow.size, ucol.size), dtype=np.float64)
    cost[ri, ci] = v
    r, c = linear_sum_assignment(-cost)
    matched = cost[r, c]
    matched = matched[matched >= tau]
    return int(matched.size), matched


# --------------------------------------------------------------------------- #
# metric assembly
# --------------------------------------------------------------------------- #
def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    denom = precision + recall
    if np.isnan(precision) or np.isnan(recall):
        f1 = np.nan
    elif denom == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / denom
    ap = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    return precision, recall, f1, ap


def _metrics_from_cont(cont, taus):
    gt_ids, pred_ids, iou, area_gt, area_pred = _iou_pairs(cont)

    n_gt = int((area_gt[1:] > 0).sum())
    n_pred = int((area_pred[1:] > 0).sum())

    # pixel / semantic level (foreground = any label > 0)
    inter_fg = float(cont[1:, 1:].sum())
    fg_gt = float(area_gt[1:].sum())
    fg_pred = float(area_pred[1:].sum())
    union_fg = fg_gt + fg_pred - inter_fg
    pixel_iou = inter_fg / union_fg if union_fg else np.nan
    dice = 2 * inter_fg / (fg_gt + fg_pred) if (fg_gt + fg_pred) else np.nan

    rows = {}
    for tau in taus:
        tp, matched_iou = _match(gt_ids, pred_ids, iou, tau)
        fp, fn = n_pred - tp, n_gt - tp
        precision, recall, f1, ap = _prf(tp, fp, fn)
        rows[tau] = dict(
            tp=tp, fp=fp, fn=fn,
            precision=precision, recall=recall, f1=f1, ap=ap,
            # extensive quantity: needed to pool mean_matched_iou / pq exactly
            sum_matched_iou=float(matched_iou.sum()),
            mean_matched_iou=float(matched_iou.mean()) if matched_iou.size else np.nan,
            # panoptic quality = seg. quality * detection quality
            pq=float(matched_iou.sum()) / (tp + 0.5 * fp + 0.5 * fn)
            if (tp + fp + fn) else np.nan,
        )

    common = dict(
        n_gt=n_gt, n_pred=n_pred,
        pixel_iou=pixel_iou, dice=dice,
        # extensive quantities: needed to pool pixel_iou / dice exactly
        inter_fg=int(inter_fg),
        fg_voxels_gt=int(fg_gt), fg_voxels_pred=int(fg_pred),
    )
    return rows, common


def get_comparison_summary(
    labels_pred,
    labels_gt,
    taus=(0.5, 0.6, 0.7, 0.75, 0.8, 0.9),
    per_slice=True,
):
    """Compare a predicted instance volume against a corrected (GT) volume.

    Parameters
    ----------
    labels_pred : (Z, Y, X) int array   — Cellpose output, 3D-stitched
    labels_gt   : (Z, Y, X) int array   — manually corrected labels (truth)
    taus        : IoU thresholds for instance matching
    per_slice   : also return a per-z dataframe

    Returns
    -------
    sample_metrics : pd.DataFrame, one row per tau (volume-level, 3D instances)
    per_slice_df   : pd.DataFrame, one row per (z, tau), or None
    """
    labels_pred = np.asarray(labels_pred)
    labels_gt = np.asarray(labels_gt)
    if labels_pred.shape != labels_gt.shape:
        raise ValueError(f"shape mismatch: {labels_pred.shape} vs {labels_gt.shape}")
    if labels_pred.ndim != 3:
        raise ValueError("expected 3D (Z, Y, X) label volumes")

    taus = tuple(taus)
    cont_total, cont_slices, _, _ = _contingency(labels_gt, labels_pred, per_slice)

    # ---- volume level (true 3D instances) ----
    rows, common = _metrics_from_cont(cont_total, taus)
    sample_metrics = pd.DataFrame(
        [{"iou_threshold": t, **common, **rows[t]} for t in taus]
    )

    # ---- per slice (2D cross-sections of the 3D labels) ----
    per_slice_df = None
    if per_slice:
        recs = []
        for z, m in enumerate(cont_slices):
            r, c = _metrics_from_cont(m, taus)
            for t in taus:
                recs.append({"z": z, "iou_threshold": t, **c, **r[t]})
        per_slice_df = pd.DataFrame(recs)

    return sample_metrics, per_slice_df


# --------------------------------------------------------------------------- #
# pooling across samples
# --------------------------------------------------------------------------- #
_MACRO_METRICS = (
    "precision", "recall", "f1", "ap",
    "pixel_iou", "dice", "mean_matched_iou", "pq",
)


def pool_summaries(summary_df, group_col="iou_threshold"):
    """Global metrics across samples, one row per IoU threshold.

    Micro-average (plain names, e.g. `f1`): TP/FP/FN summed across samples,
    ratios recomputed once. Each *nucleus* carries equal weight, so
    nucleus-rich samples dominate.

    Macro-average (`*_macro`) with SD (`*_std`, ddof=1): mean and spread of the
    per-sample values. Each *sample* carries equal weight.

    IMPORTANT for reporting: an SD is a spread *across samples*, so a
    "mean ± SD" string must pair `*_macro` with `*_std`. Do not write
    micro ± std -- the halves come from different estimators. `format_report()`
    enforces this.
    """
    out = []
    for tau, g in summary_df.groupby(group_col, sort=True):
        tp, fp, fn = int(g.tp.sum()), int(g.fp.sum()), int(g.fn.sum())
        precision, recall, f1, ap = _prf(tp, fp, fn)

        inter = float(g.inter_fg.sum())
        fg_gt = float(g.fg_voxels_gt.sum())
        fg_pred = float(g.fg_voxels_pred.sum())
        union = fg_gt + fg_pred - inter
        smi = float(g.sum_matched_iou.sum())

        rec = {
            group_col: tau,
            "n_samples": int(g.shape[0]),
            "n_gt": int(g.n_gt.sum()),
            "n_pred": int(g.n_pred.sum()),
            "tp": tp, "fp": fp, "fn": fn,
            # --- micro ---
            "precision": precision, "recall": recall, "f1": f1, "ap": ap,
            "pixel_iou": inter / union if union else np.nan,
            "dice": 2 * inter / (fg_gt + fg_pred) if (fg_gt + fg_pred) else np.nan,
            "mean_matched_iou": smi / tp if tp else np.nan,
            "pq": smi / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else np.nan,
        }

        # --- macro mean + across-sample SD ---
        n = g.shape[0]
        for m in _MACRO_METRICS:
            if m in g.columns:
                rec[f"{m}_macro"] = g[m].mean()
                rec[f"{m}_std"] = g[m].std(ddof=1) if n > 1 else np.nan

        out.append(rec)
    return pd.DataFrame(out)


def format_report(global_df, tau=0.5, decimals=2):
    """Manuscript-ready 'mean ± SD' strings for one IoU threshold.

    Always pairs the macro mean with the across-sample SD, so both halves of
    each '±' come from the same estimator. The SD is over samples
    (n = n_samples), NOT a confidence interval -- say so in the caption.
    """
    row = global_df.loc[global_df.iou_threshold == tau]
    if row.empty:
        raise ValueError(f"tau={tau} not present; have {list(global_df.iou_threshold)}")
    r = row.iloc[0]

    def pm(name, d=decimals):
        mu, sd = r.get(f"{name}_macro", np.nan), r.get(f"{name}_std", np.nan)
        if np.isnan(mu):
            return "n/a"
        if np.isnan(sd):
            return f"{mu:.{d}f} (n=1)"
        return f"{mu:.{d}f} ± {sd:.{d}f}"

    return "\n".join([
        f"n = {int(r.n_samples)} samples, {int(r.n_gt)} annotated nuclei, IoU >= {tau}",
        f"  F1         {pm('f1')}     (micro {r.f1:.3f})",
        f"  precision  {pm('precision')}     (micro {r.precision:.3f})",
        f"  recall     {pm('recall')}     (micro {r.recall:.3f})",
        f"  pixel IoU  {pm('pixel_iou')}     (micro {r.pixel_iou:.3f})",
        f"  Dice       {pm('dice')}     (micro {r.dice:.3f})",
        "",
        "  +/- = SD across samples (ddof=1). Means are macro-averaged (each",
        "  sample weighted equally); micro values shown for reference.",
    ])


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from skimage import io

    dir_labels = (
        "data/segmentation_validation_inhibitors/"
        "004/004-01/labels"
    )
    samples = ["B08", "C08", "D08"]

    all_summaries, all_slices = [], []
    for sample in samples:
        labels_init = io.imread(f"{dir_labels}/predicted/{sample}.tif")
        labels_corr = io.imread(f"{dir_labels}/corrected/restitched_{sample}.tif")

        sample_metrics, per_slice_df = get_comparison_summary(
            labels_pred=labels_init,
            labels_gt=labels_corr,
        )
        sample_metrics.insert(0, "sample", sample)
        per_slice_df.insert(0, "sample", sample)

        all_summaries.append(sample_metrics)
        all_slices.append(per_slice_df)

        print(f"\n=== {sample} ===")
        print(
            sample_metrics[
                ["iou_threshold", "n_gt", "n_pred", "tp", "fp", "fn",
                 "precision", "recall", "f1", "pixel_iou", "dice"]
            ].to_string(index=False)
        )

        del labels_init, labels_corr

    summary = pd.concat(all_summaries, ignore_index=True)
    slices = pd.concat(all_slices, ignore_index=True)
    global_metrics = pool_summaries(summary)

    print(f"\n=== GLOBAL (pooled over {len(samples)} samples) ===")
    print(
        global_metrics[
            ["iou_threshold", "tp", "fp", "fn",
             "precision_macro", "precision_std",
             "recall_macro", "recall_std",
             "f1_macro", "f1_std"]
        ].to_string(index=False)
    )

    print("\n" + format_report(global_metrics, tau=0.5))

    summary.to_csv(f"{dir_labels}/segmentation_summary.csv", index=False)
    slices.to_csv(f"{dir_labels}/segmentation_per_slice.csv", index=False)
    global_metrics.to_csv(f"{dir_labels}/segmentation_global.csv", index=False)

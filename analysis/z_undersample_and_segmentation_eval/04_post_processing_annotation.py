import numpy as np
from cellpose.utils import stitch3D
from skimage import io
import pandas as pd
from skimage.measure import label as cc_label
from pathlib import Path
from tqdm import tqdm

def make_labels_unique(labels, background=0, connectivity=2):
    out = labels.copy()
    next_label = int(labels.max()) + 1

    for val in np.unique(labels):
        if val == background:
            continue
        cc = cc_label(labels == val, connectivity=connectivity)
        if cc.max() <= 1:
            continue                      # already one region, leave it
        for i in range(2, cc.max() + 1):  # keep first blob, relabel the rest
            out[cc == i] = next_label
            next_label += 1
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Directory to save output")
    args = parser.parse_args()

    file = args.file
    print(f"file: {file}")
    path_stem = Path(file).stem
    file_out = Path(Path(file).parent, f"restitched_{path_stem}{Path(file).suffix}")
    print(f"file_out: {file_out}")
    labels = io.imread(file)
    print(labels.dtype)
    labels = np.stack(
        [make_labels_unique(labels[z]) for z in tqdm(range(labels.shape[0]), desc="Processing slices")],
        axis=0,
    )
    print(labels.shape)
    labels = stitch3D(labels)
    print(labels.shape)
    labels = labels.astype(np.uint16)
    io.imsave(Path(file_out), labels)

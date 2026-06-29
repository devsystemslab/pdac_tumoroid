from pathlib import Path

import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import pandas as pd
import scanpy as sc
import skimage
import yaml
from analysis.pilotscreen.generate_example_images import (
    order_genes,
    plot_dotplot,
    plot_organoid,
    plot_umap,
)
from analysis.timecourse.correct_examples import add_scalebar
from matplotlib import cm
from skimage import io

from image_processing.montage import load_well, rgb_overlay
from image_processing.utils import get_metadata, scale_image

screen = "pilotscreen"
file = "/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/configs/params.yaml"

with open(file) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params[screen]

dir_screen = params["dir_screen"]
dir_results = Path(params["dir_screen"], "anndata")

# ORIGINAL LABELS

cycles = ["01", "03"]
plates = ["004", "005"]
mods_vertical = []
for c in cycles:
    mdata_orig = mu.read_h5mu(Path(dir_results, f"mdata_cycle-{c}.h5mu"))
    plate_horizontal = []
    for plate in plates:
        print(m)
        img_1 = plot_organoid(
            f"J08_{plate}",
            mdata_orig["phenocoder"],
            project_3d=True,
            cut_open=True,
            bg_color="white",
            edgecolors="#d3d3d3",
            add_legend=False,
        )
        plt.imshow(img_1)
        plt.show()
        plate_horizontal.append(img_1)
    img_horizontal = np.hstack(plate_horizontal)
    plt.imshow(img_horizontal)

    mods_vertical.append(img_horizontal)
img_full = np.vstack(mods_vertical)
plt.imshow(img_full)
plt.axis("off")
plt.savefig(
    f"example_wells_orig_labels.png",
    bbox_inches="tight",
    pad_inches=0,
    format="png",
    dpi=500,
)
plt.show()

# NEW PROJECTED LABELS
mdata_projected = mu.read_h5mu(Path(dir_results, "mdata_plate4_projected.h5mu"))

cycles = ["cycle01_phenocoder", "cycle03_phenocoder"]
plates = ["004", "005"]
mods_vertical = []
for c in cycles:
    plate_horizontal = []
    for plate in plates:
        print(m)
        img_1 = plot_organoid(
            f"J08_{plate}",
            mdata_projected[c],
            project_3d=True,
            cut_open=True,
            bg_color="white",
            edgecolors="#d3d3d3",
            add_legend=False,
        )
        plt.imshow(img_1)
        plt.show()
        plate_horizontal.append(img_1)
    img_horizontal = np.hstack(plate_horizontal)
    plt.imshow(img_horizontal)

    mods_vertical.append(img_horizontal)
img_full = np.vstack(mods_vertical)
plt.imshow(img_full)
plt.axis("off")
plt.savefig(
    f"example_wells_projected_labels.png",
    bbox_inches="tight",
    pad_inches=0,
    format="png",
    dpi=500,
)
plt.show()

import os
from math import cos, radians, sin, sqrt
from pathlib import Path

import anndata as ad
import muon as mu
import numpy as np
import pandas as pd
import yaml
from skimage import io
from skimage.exposure import rescale_intensity
from tqdm import tqdm

from whole_mount_tumoroid.image_processing.utils import get_metadata
from whole_mount_tumoroid.phenocoder.utils import scale_image


class RGBRotate:
    """Rotate RGB colors in hue space."""

    def __init__(self):
        self.matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def set_hue_rotation(self, degrees):
        """Set hue rotation in degrees."""
        cosA = cos(radians(degrees))
        sinA = sin(radians(degrees))
        self.matrix[0][0] = cosA + (1.0 - cosA) / 3.0
        self.matrix[0][1] = 1.0 / 3.0 * (1.0 - cosA) - sqrt(1.0 / 3.0) * sinA
        self.matrix[0][2] = 1.0 / 3.0 * (1.0 - cosA) + sqrt(1.0 / 3.0) * sinA
        self.matrix[1][0] = 1.0 / 3.0 * (1.0 - cosA) + sqrt(1.0 / 3.0) * sinA
        self.matrix[1][1] = cosA + (1.0 - cosA) / 3.0
        self.matrix[1][2] = 1.0 / 3.0 * (1.0 - cosA) - sqrt(1.0 / 3.0) * sinA
        self.matrix[2][0] = 1.0 / 3.0 * (1.0 - cosA) - sqrt(1.0 / 3.0) * sinA
        self.matrix[2][1] = 1.0 / 3.0 * (1.0 - cosA) + sqrt(1.0 / 3.0) * sinA
        self.matrix[2][2] = cosA + (1.0 - cosA) / 3.0

    def apply(self, r, g, b):
        """Apply hue rotation to RGB values."""
        r_new = self.matrix[0][0] * r + self.matrix[0][1] * g + self.matrix[0][2] * b
        g_new = self.matrix[1][0] * r + self.matrix[1][1] * g + self.matrix[1][2] * b
        b_new = self.matrix[2][0] * r + self.matrix[2][1] * g + self.matrix[2][2] * b
        return (
            int(np.clip(r_new, 0, 255)),
            int(np.clip(g_new, 0, 255)),
            int(np.clip(b_new, 0, 255)),
        )


def order_dataframe_hierarchical(
    df: pd.DataFrame,
    order_cancer: list,
    order_caf: list,
    order_drugs: list,
    cancer_col: str = "cancer",
    caf_col: str = "caf",
    drug_col: str = "drug",
) -> pd.DataFrame:
    """
    Order dataframe hierarchically by cancer patient ID, CAF patient ID, and drug treatment.

    :param df: DataFrame to sort
    :param order_cancer: List of cancer patient IDs in desired order
    :param order_caf: List of CAF patient IDs in desired order
    :param order_drugs: List of drug treatments in desired order
    :param cancer_col: Column name for cancer patient ID (default: "cancer")
    :param caf_col: Column name for CAF patient ID (default: "caf")
    :param drug_col: Column name for drug treatment (default: "drug")
    :return: Sorted DataFrame
    """
    # Create categorical types with specified order
    df = df.copy()

    df[cancer_col] = pd.Categorical(
        df[cancer_col], categories=order_cancer, ordered=True
    )
    df[caf_col] = pd.Categorical(df[caf_col], categories=order_caf, ordered=True)
    df[drug_col] = pd.Categorical(df[drug_col], categories=order_drugs, ordered=True)

    # Sort by cancer, then caf, then drug
    df = df.sort_values(by=[cancer_col, caf_col, drug_col])

    return df


def plot_for_montage(well_id: str):
    """
    Extract or generate a plot for a given well.

    :param well_id: Well identifier
    :return: Image array for the plot, or None if not available
    """
    # This is a placeholder - implement according to your plotting needs
    # Example: load from a saved plot or generate one dynamically
    try:
        # Implement your plot generation/loading logic here
        # For now, return None
        return None
    except Exception as e:
        print(f"Failed to generate plot for {well_id}: {e}")
        return None


def read_image_for_well(
    well_id: str,
    df_images: pd.DataFrame = None,
    lut_dict: dict = None,
    n_down_sampling: int = 8,
) -> np.ndarray:
    """
    Read and process an image for a given well with multicolor overlay.

    :param well_id: Well identifier (e.g., 'C04_008')
    :param df_images: DataFrame containing image metadata with columns:
                      'well_id', 'plate_id', 'file_path', 'cycle'
    :param lut_dict: Dictionary with cycle -> list of (percentile_min, percentile_max) tuples
    :param n_down_sampling: Downsampling factor (default: 4)
    :return: Processed RGB image array, or None if not found
    """
    if df_images is None:
        print(f"df_images is required to read images")
        return None

    if lut_dict is None:
        lut_dict = {
            "01": [(1, 99), (1, 99), (1, 99), (1, 99)],
            "03": [(5, 95), (5, 95), (95, 99), (5, 95)],
        }

    # Extract well and plate from well_id (e.g., 'C04_008' -> 'C04', '008')
    parts = well_id.split("_")
    if len(parts) != 2:
        print(f"Invalid well_id format: {well_id}")
        return None

    well = parts[0]
    plate = parts[1]

    # Find matching images in df_images
    df_match = df_images[
        (df_images["well_id"] == well) & (df_images["plate_id"] == plate)
    ]

    if df_match.empty:
        print(f"No images found for well {well_id}")
        return None

    # Extract cycle from dir_images path
    dir_images = df_match.iloc[0]["dir_images"]
    cycle = dir_images.split("-")[-1].split("/")[0]

    # Sort by channel_id to ensure proper channel order
    df_match["channel_id"] = df_match["channel_id"].astype(int)
    df_match = df_match.sort_values("channel_id")

    # Read and process each channel
    imgs = []
    for idx, (_, row) in enumerate(df_match.iterrows()):
        file_name = row["file"]
        file_path = os.path.join(row["dir_images"], file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        img = io.imread(file_path)
        # Apply lookup table (LUT) for contrast adjustment
        if idx < len(lut_dict[cycle]):
            range_lut = (
                np.percentile(img, lut_dict[cycle][idx][0]),
                np.percentile(img, lut_dict[cycle][idx][1]),
            )
            img = rescale_intensity(img, in_range=range_lut, out_range=(0, 1.0))
        imgs.append(img)

    if len(imgs) == 0:
        print(f"No valid images found for {well_id}")
        return None

    imgs = np.asarray(imgs)
    # Downsample pixels
    if n_down_sampling is not None:
        imgs = np.asarray([img[::n_down_sampling, ::n_down_sampling] for img in imgs])

    # Move channel axis to last position
    imgs = np.moveaxis(imgs, 0, -1)

    # Create multicolor overlay
    # Use channels 1:3 as RGB, blend with channel 0
    img_rgb = imgs[..., 1:]
    img_ch0 = imgs[..., 0]

    # Duplicate channel 0 to 3 channels and blend
    img_ch0_rgb = np.repeat(img_ch0[:, :, np.newaxis], 3, axis=2)
    img = (img_rgb + img_ch0_rgb) / 2

    # Rescale to 0-255 and convert to uint8
    img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)

    # Apply hue rotation for better color visualization
    rotator = RGBRotate()
    rotator.set_hue_rotation(49)
    img = (
        np.asarray([rotator.apply(r, g, b) for r, g, b in img.reshape(-1, 3)])
        .reshape(img.shape)
        .astype(np.uint8)
    )
    img = scale_image(img, range=(0, 255)).astype(np.uint8)

    return img


if __name__ == "__main__":
    file = "/pstore/data/ihb-tumoroid/data/processed/egfr/anndata/mdata_subset.h5mu"
    mdata = mu.read_h5mu(file)
    adata = mdata["phenocoder_combined"].copy()
    file_wells = "/pstore/home/harmelc/tumoroid_screen/whole_mount_tumoroid/analysis/egfr/well_example.yaml"
    dict_wells = yaml.safe_load(open(file_wells))
    wells = [f"{well}_{key}" for key in dict_wells for well in dict_wells[key]]
    df = adata.obs[adata.obs.index.isin(wells)]
    df["cancer_caf_treatment"] = (
        df["cancer"].astype(str)
        + "_"
        + df["caf"].astype(str)
        + "_"
        + df["drug"].astype(str)
    )
    # sample one from each cancer_caf_treatment
    df = df.groupby("cancer_caf_treatment").sample(1)
    # arrange by patients
    order_patients = ["P382", "P388", "P506", "P585"]
    drug_order = ["DMSO", "Erlotinib", "Osimertinib"]
    # arrange columnns cancer, caf by order patient and then in combination with drug order
    df = order_dataframe_hierarchical(df, order_patients, order_patients, drug_order)
    df["cancer_x_caf"] = df["cancer"].astype(str) + "_" + df["caf"].astype(str)
    dir_images = "/pstore/data/ihb-tumoroid/data/processed/egfr"
    # get df_images
    df_images = pd.concat(
        [
            get_metadata(f"{dir_images}/008/008-01/TIF_MIP_OVR_BG").assign(
                plate_id="008"
            ),
            get_metadata(f"{dir_images}/009/009-01/TIF_MIP_OVR_BG").assign(
                plate_id="009"
            ),
        ]
    )

    dir_out = "/pstore/home/harmelc/tumoroid_screen/whole_mount_tumoroid/analysis/egfr/plots/example_images"
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    df["file_img"] = df.index.map(lambda i: os.path.join(dir_out, f"{i}.png"))
    for i in tqdm(df.index, desc="Generating example images"):
        img = read_image_for_well(i, df_images)
        io.imsave(df.loc[i, "file_img"], img)
    dir_out_montages = os.path.join(dir_out, "montages")
    Path(dir_out_montages).mkdir(parents=True, exist_ok=True)
    for cancer_x_caf in tqdm(df["cancer_x_caf"].unique()):
        df_tmp = df[df["cancer_x_caf"] == cancer_x_caf]
        if len(df_tmp) != len(drug_order):
            print(f"Skipping {cancer_x_caf} (len={len(df_tmp)} != {len(drug_order)})")
            continue
        imgs = []
        for drug in drug_order:
            file = df_tmp[df_tmp["drug"] == drug]["file_img"].values[0]
            img = io.imread(file)
            imgs.append(img)
        img = np.hstack(imgs)
        io.imsave(os.path.join(dir_out_montages, f"{cancer_x_caf}.png"), img)

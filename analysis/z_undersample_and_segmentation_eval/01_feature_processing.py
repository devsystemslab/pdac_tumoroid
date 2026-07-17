from pathlib import Path

import numpy as np
import pandas as pd
from cellpose.utils import stitch3D
from skimage import io
from skimage.measure import regionprops_table
from tqdm import tqdm

from image_processing.utils import get_metadata


def process_well_features(well, z_stack, dir_images, dir_segmented):

    df_images = get_metadata(dir_images)
    # load metadata
    df_images = df_images[(df_images["well_id"] == well) & (df_images["z_stack_id"] == z_stack)]
    # arrange by imaging cycle and channel
    df_images = df_images.sort_values(by="channel_id")

    # get metadata for label images
    df_label_images = get_metadata(dir_segmented)
    df_label_images = df_label_images[(df_label_images["z_stack_id"] == z_stack) & (df_label_images["well_id"] == well)]

    if df_label_images.shape[0] == 0:
        return []

    assert df_label_images.shape[0] == 1

    # load images
    img_label = io.imread(Path(df_label_images["dir_images"].iloc[0], df_label_images["file"].iloc[0]))

    imgs = np.asarray(
        [io.imread(Path(df_images["dir_images"].iloc[i], df_images["file"].iloc[i])) for i in range(df_images.shape[0])]
    )

    df = pd.DataFrame(
        regionprops_table(
            label_image=img_label,
            intensity_image=np.moveaxis(imgs, 0, -1),
            properties=(
                "label",
                "centroid",
                "area",
                "eccentricity",
                "intensity_mean",
                "major_axis_length",
                "minor_axis_length",
            ),
        )
    )
    # add metadata
    df["well"] = well
    df["z_stack"] = z_stack

    return df


def process_features_restitch(well, dir_images, dir_segmented):
    df_images = get_metadata(dir_images)
    df_images = df_images[(df_images["well_id"] == well) & (df_images["channel_id"] == "01")]
    df_images = df_images.sort_values(by="channel_id")
    df_images["z_stack_id"] = df_images["z_stack_id"].astype(int)
    df_images = df_images.sort_values(by="z_stack_id")
    # get metadata for label images
    df_label_images = get_metadata(dir_segmented)
    df_label_images = df_label_images[df_label_images["well_id"] == well]
    df_label_images["z_stack_id"] = df_label_images["z_stack_id"].astype(int)
    df_label_images = df_label_images.sort_values(by="z_stack_id")

    # load images
    img_label = np.asarray(
        [
            io.imread(
                Path(
                    df_label_images["dir_images"].iloc[i],
                    df_label_images["file"].iloc[i],
                )
            )
            for i in tqdm(
                range(df_label_images.shape[0]),
                desc=f"{well} - Loading label images",
                total=df_label_images.shape[0],
            )
        ]
    )

    imgs = np.asarray(
        [
            io.imread(Path(df_images["dir_images"].iloc[i], df_images["file"].iloc[i]))
            for i in tqdm(
                range(df_images.shape[0]),
                desc=f"{well} - Loading images",
                total=df_images.shape[0],
            )
        ]
    )

    df = []
    for i in tqdm(range(1, 10), desc=f"{well} - Feature extraction"):
        imgs_tmp = imgs[::i, :, :]
        imgs_label_tmp = img_label[::i, :, :]
        # re do label merging
        if i != 1:
            imgs_label_tmp = stitch3D(imgs_label_tmp)
        for j in range(imgs_tmp.shape[0]):
            df_tmp = pd.DataFrame(
                regionprops_table(
                    label_image=imgs_label_tmp[j],
                    intensity_image=imgs_tmp[j],
                    properties=(
                        "label",
                        "centroid",
                        "area",
                        "eccentricity",
                        "intensity_mean",
                        "major_axis_length",
                        "minor_axis_length",
                    ),
                )
            )
            df_tmp["n_skip"] = i
            df_tmp["z_slice_orig"] = j * i
            df_tmp["z_slice_skip"] = j
            df.append(df_tmp)
    df = pd.concat(df)
    return df


if __name__ == "__main__":
    dir_data = "data/segmentation_validation_inhibitors/004/004-01"
    dir_images = Path(dir_data, "TIF_OVR_BG")
    dir_segmented = Path(dir_data, "SEG_TIF_OVR_BG")
    df_samples = get_metadata(dir_segmented)
    samples = df_samples["well_id"].unique()

    # assess undersampling from complete 3d stitched labels
    Path(dir_data, "segmentation_plots").mkdir(exist_ok=True)
    for sample in samples:
        df_features = []
        z_slices = df_samples[(df_samples["well_id"] == sample)]["z_stack_id"].unique()
        for z_slice in tqdm(z_slices, desc=f"Processing {sample}"):
            df_features.append(process_well_features(sample, z_slice, dir_images, dir_segmented))
        df_features = pd.concat(df_features, axis=0)
        df_features.to_csv(Path(dir_data, "segmentation_plots", f"segmentation_features_{sample}.csv"))
        df_features.to_csv(Path(dir_data, "segmentation_plots", f"segmentation_features_{sample}.csv"))

    # # assess undersampling with restitching labels
    df_features = []
    for sample in samples:
        df_features.append(process_features_restitch(sample, dir_images, dir_segmented))
    df_features = pd.concat(df_features, axis=0)
    df_features.to_csv(Path(dir_data, "segmentation_plots", f"segmentation_features_restitched.csv"))

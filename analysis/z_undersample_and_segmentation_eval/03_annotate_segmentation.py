import os
from pathlib import Path

import dask.array as da
import napari
import numpy as np
import pandas as pd
from cellpose.utils import stitch3D
from dask.array.image import imread
from skimage import io
from skimage.measure import regionprops
from tqdm import tqdm


def get_metadata(dir_images):
    """
    Get metadata from image filenames
    :param dir_images:
    :return:
    """
    images = os.listdir(dir_images)
    images = [image for image in images if ".tif" in image]
    if len(images) > 0:
        regex = r"_(?P<well_id>[A-Z]\d{2})_T(?P<time_point>\d{4})F(?P<field_id>\d{3})L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})Z(?P<z_stack_id>\d{2,3})C(?P<channel_id>\d{2})\.tif$"
        df = pd.DataFrame({"file": images, "dir_images": str(dir_images)})
        df = df.join(df["file"].str.extractall(regex).groupby(level=0).last())
        # remove rows that have nan in any column
        df = df[~df.isna().any(axis=1)]
        return df
    else:
        return pd.DataFrame().assign(
            well_id=[None],
            time_point=[None],
            field_id=[None],
            time_line_id=[None],
            action_id=[None],
            z_stack_id=[None],
            channel_id=[None],
        )


def remove_labels(imgs, labels_to_remove):
    """Set specified labels to 0 (background) using a LUT."""
    if len(labels_to_remove) == 0:
        return imgs
    lut = np.arange(imgs.max() + 1, dtype=imgs.dtype)
    lut[np.asarray(labels_to_remove, dtype=lut.dtype)] = 0
    return lut[imgs]


def load_images(
    df_images,
    well,
    channel,
    sampling=1,
    z_step_um=10.0,
    xy_pixel_size_um=0.322,
    dropout=None,
    restitch_labels=False,
    filter_labels=False,
):
    """
    Load images for a given well and channel
    :param df_images:
    :param well:
    :param channel:
    :param sampling:
    :param z_step_um: Z-step size in micrometers (default 10 µm)
    :param xy_pixel_size_um: XY pixel size in micrometers (default 0.322 µm/px)
    :return: tuple of (imgs, scale) where scale is the voxel size for napari
    """
    if well is None:
        print("No well selected.")
        return None
    # filter images and sort by z_stack_id
    df_images = df_images[
        (df_images["well_id"] == well) & (df_images["channel_id"] == channel)
    ]
    df_images["z_stack_id"] = df_images["z_stack_id"].astype(int)
    df_images = df_images.sort_values(by=["z_stack_id"], ascending=True)
    if dropout is not None:
        df_images = df_images.iloc[::dropout]
        z_step_um = z_step_um * dropout
    # load images
    n_images = len(df_images)
    desc = f"well {well} ch{channel}"
    if n_images > 1:
        imgs = [
            imread(str(Path(row["dir_images"], row["file"])))
            for index, row in tqdm(df_images.iterrows(), desc=desc, total=n_images)
        ]
        imgs = da.stack(imgs)
        # Squeeze out single channel dimension if present (shape: z, c, y, x -> z, y, x)
        if imgs.shape[1] == 1:
            imgs = imgs.squeeze(axis=1)
        if filter_labels:
            # remove labels that appear on just 1 z-slice
            label_regions = regionprops(imgs)
            labels_to_remove = [
                region.label
                for region in tqdm(label_regions)
                if region.bbox[3] - region.bbox[0] == 1  # z-extent of 1 slice
            ]
            imgs = remove_labels(imgs, labels_to_remove)
        if restitch_labels:
            # convert to numpy array
            imgs = imgs.compute()
            imgs = stitch3D(imgs)
    else:
        imgs = [
            io.imread(str(Path(row["dir_images"], row["file"])))
            for index, row in tqdm(df_images.iterrows(), desc=desc, total=n_images)
        ]
        imgs = np.asarray(imgs)
        # Squeeze out single channel dimension if present
        if imgs.shape[1] == 1:
            imgs = imgs.squeeze(axis=1)
    if sampling > 1:
        imgs = imgs[..., ::sampling, ::sampling]
        # Adjust scale for sampling
        scale = (z_step_um, xy_pixel_size_um * sampling, xy_pixel_size_um * sampling)
    else:
        scale = (z_step_um, xy_pixel_size_um, xy_pixel_size_um)
    return imgs, scale


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load images for a given well and channel"
    )
    parser.add_argument("--dir_images", type=str, help="Directory containing images")
    parser.add_argument(
        "--dir_labels", type=str, help="Directory containing annotations"
    )
    parser.add_argument("--well", type=str, help="Well ID")
    parser.add_argument("--channel", type=str, default="01", help="Channel ID")
    parser.add_argument("--sampling", type=int, default=1, help="Sampling factor")
    parser.add_argument(
        "--z_step_um", type=float, default=10.0, help="Z-step size in micrometers"
    )
    parser.add_argument(
        "--xy_pixel_size_um",
        type=float,
        default=0.322,
        help="XY pixel size in micrometers",
    )
    parser.add_argument("--dropout", type=int, default=None, help="Dropout factor")
    parser.add_argument(
        "--restitch_labels", action="store_true", help="Restitch labels"
    )
    parser.add_argument(
        "--filter_labels", action="store_true", help="Filter labels on 1-slice extent"
    )
    args = parser.parse_args()
    if args.well is None:
        raise ValueError("Well ID is required")
    dir_images = args.dir_images
    df_images = get_metadata(args.dir_images)
    imgs, scale = load_images(
        df_images,
        args.well,
        args.channel,
        args.sampling,
        z_step_um=args.z_step_um,
        xy_pixel_size_um=args.xy_pixel_size_um,
        dropout=args.dropout,
    )
    df_labels = get_metadata(args.dir_labels)
    labels, label_scales = load_images(
        df_labels,
        args.well,
        args.channel,
        args.sampling,
        z_step_um=args.z_step_um,
        xy_pixel_size_um=args.xy_pixel_size_um,
        dropout=args.dropout,
        restitch_labels=args.restitch_labels,
        filter_labels=args.filter_labels,
    )
    viewer = napari.Viewer()
    viewer.add_image(imgs, scale=scale, name="image")
    viewer.add_image(labels, scale=label_scales, name="label_image")
    napari.run()

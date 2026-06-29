import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage import io
from skimage.util import montage
from tqdm import tqdm

from image_processing.utils import get_metadata, scale_image


def get_plate_intensity_statistics(dir_images: str):
    """
    Get summary statistics for all images file in dir_images.
    :param dir_images:
    :return:
    """
    df_images = get_metadata(dir_images)
    # process images
    df_summary = []
    imgs = []
    for i, row in tqdm(df_images.iterrows(), total=df_images.shape[0]):
        if not Path(dir_images, row["file"]).exists():
            continue
        df = row.copy()
        img = io.imread(Path(dir_images, row["file"]))
        imgs.append(img)
        df["max"] = np.max(img)
        df["mean"] = np.mean(img)
        df["std"] = np.std(img)
        df["min"] = np.min(img)
        df_summary.append(df)
    df_summary = pd.concat(df_summary, axis=1).T
    imgs = np.asarray(
        [scale_image(img[::8, ::8], range=(0, 255)).astype(np.uint8) for img in imgs]
    )
    return df_summary, imgs


def run_qc(dir_images: str, plot: True):
    """
    Run qc.
    :param dir_images:
    :param plot:
    :return:
    """
    df_summary, imgs = get_plate_intensity_statistics(dir_images)
    img_montage = montage(imgs, grid_shape=(16, 24))
    io.imsave(Path(dir_images, f"montage_all_images.png"), img_montage)
    df_summary.to_csv(Path(dir_images, "qc_intensities.csv"), index=False)
    # aggregate
    df_agg = (
        df_summary[["channel_id", "max", "mean", "std", "min"]]
        .groupby("channel_id")
        .mean()
        .reset_index()
    )
    df_agg.to_csv(Path(dir_images, "qc_intensities_agg.csv"), index=False)
    if plot:
        for channel in df_agg["channel_id"].unique():
            df_plot = df_summary[df_summary["channel_id"] == channel]
            df_plot_agg = df_agg[df_agg["channel_id"] == channel]
            plt.figure()
            sns.pairplot(df_plot)
            plt.savefig(Path(dir_images, f"qc_pairplot_channel-{channel}.png"))

            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax = sns.histplot(df_plot, x="mean")
            ax.axvline(
                x=df_plot_agg["mean"].values[0],
                c="red",
                ymin=0,  # Bottom of the plot
                ymax=1,
            )  # Top o
            fig.savefig(
                Path(dir_images, f"qc_hist_mean_intensities_channel-{channel}.png")
            )
            plt.close("all")
    return df_agg


if __name__ == "__main__":
    dir_yokogawa = "InstrumentData/YokogawaCV8000_Optimus_prime"
    datasets = [
        dataset
        for dataset in os.listdir(dir_yokogawa)
        if os.path.isdir(Path(dir_yokogawa, dataset))
    ]
    df_stats = []
    for dataset in datasets:
        dirs = os.listdir(Path(dir_yokogawa, dataset))
        for path_images in dirs:
            if path_images.endswith("_sf"):
                print(path_images)
                df_agg = run_qc(Path(dir_yokogawa, dataset, path_images), plot=True)
                df_stats.append(
                    df_agg.assign(plate_id=dataset, path_images=path_images)
                )
            else:
                continue

    df_stats = pd.concat(df_stats)
    df_stats["dir_images"] = f"{dir_yokogawa}/" + df_stats["plate_id"]
    df_meta = pd.read_csv("whole_mount_tumoroid/metafiles/imaging_data_overview.csv")
    df_merged = pd.merge(df_stats, df_meta, on="dir_images", how="left")
    df_merged.to_csv(Path(dir_yokogawa, "qc_stats.csv"), index=False)
    sns.histplot(df_merged, x="mean", hue="screen")
    plt.savefig(Path(dir_yokogawa, "overview_qc.png"))

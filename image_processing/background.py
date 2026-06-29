from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from basicpy import BaSiC
from dask.distributed import as_completed
from skimage import io, morphology
from skimage.filters.rank import median
from smo import SMO

from image_processing.utils import get_metadata, setup_dask_client


def load_well(dir_plate: str, well: str, channel: str, input_type: str) -> np.ndarray:
    """
    Load well images
    :param dir_plate:
    :param well:
    :param channel:
    :param input_type:
    :return:
    """
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)
    df_images = df_images[
        (df_images["well_id"] == well) & (df_images["channel_id"] == channel)
    ].sort_values(by=["z_stack_id"], ascending=True)
    imgs = [
        io.imread(Path(row["dir_images"], row["file"]))
        for index, row in df_images.iterrows()
    ]
    imgs = np.asarray(imgs)
    return imgs


def correct_background(
    well: str,
    channel: str,
    z_slice: str,
    bg_value: int,
    dir_plate: str,
    input_type: str,
    dir_output: str,
) -> str:
    """
    Correct background
    :param well:
    :param channel:
    :param z_slice:
    :param dir_plate:
    :param input_type:
    :param bg_value:
    :param dir_output:
    :return:
    """
    df_images = get_metadata(Path(dir_plate, input_type))
    df_images = df_images[
        (df_images["well_id"] == well)
        & (df_images["channel_id"] == channel)
        & (df_images["z_stack_id"] == z_slice)
    ]
    assert df_images.shape[0] == 1
    img = io.imread(Path(df_images["dir_images"].iloc[0], df_images["file"].iloc[0]))
    if np.isnan(bg_value):
        bg_value = 0
    img = img.astype(int) - int(bg_value)
    img[img < 0] = 0

    # write corrected image
    file = Path(dir_output, df_images["file"].iloc[0])
    io.imsave(file, img.astype(np.uint16), check_contrast=False)
    return file


def get_bg_value(image: np.ndarray, smo: SMO) -> int:
    """
    Get background value from image
    :param image:
    :param smo:
    :return:
    """
    bg_mask = smo.bg_mask(image, threshold=0.05)
    bg_value = np.median(bg_mask.compressed())
    return bg_value


def analyze_well(well: str, channel: str, dir_plate: str, input_type: str) -> tuple:
    """
    Analyze well for background values
    :param well:
    :param channel:
    :param dir_plate:
    :param input_type:
    :return:
    """
    smo = SMO(sigma=0, size=7, shape=(1024, 1024))
    imgs = load_well(dir_plate, well, channel, input_type)
    median_value = [np.median(img) for img in imgs]
    bg_values = [get_bg_value(img, smo) for img in imgs]
    return bg_values, median_value


def process_well_estimate_bg(
    well: str, dir_plate: str, input_type: str, output_type: str = "background"
) -> str:
    """
    Process well for background values
    :param well:
    :param dir_plate:
    :param input_type:
    :param output_type:
    :return:
    """
    channels = ["01", "02", "03", "04"]
    df_bg = pd.DataFrame()
    for channel in channels:
        results_bg, results_median = analyze_well(well, channel, dir_plate, input_type)
        df_tmp = pd.DataFrame()
        results_bg = np.asarray(results_bg)
        results_median = np.asarray(results_median)
        df_tmp["bg_value"] = results_bg
        df_tmp["median_value"] = results_median
        df_tmp["channel"] = channel
        df_tmp["z_slice"] = np.arange(0, results_bg.shape[0])
        df_tmp["well"] = well
        df_bg = pd.concat([df_bg, df_tmp])
    # format z_slice
    df_bg["z_slice"] = df_bg["z_slice"] + 1
    df_bg["z_slice"] = df_bg["z_slice"].apply(lambda x: f"{x:02d}")
    # save file
    file = Path(dir_plate, output_type, f"df_bg_{well}_{input_type}.csv")
    df_bg.to_csv(file, index=False)
    return file


def process_well_bleach_correction(
    well: str,
    channel: str,
    median_filter: bool,
    correction_mode: str,
    dir_plate: str,
    input_type: str,
    dir_output: str,
) -> None:
    """
    Process well for bleach correction
    :param well:
    :param channel:
    :param median_filter:
    :param correction_mode:
    :param dir_plate:
    :param input_type:
    :param dir_output:

    :return:
    """
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)
    df_images = df_images[
        (df_images["well_id"] == well) & (df_images["channel_id"] == channel)
    ].sort_values(by=["z_stack_id"], ascending=True)
    files = df_images["file"].tolist()
    imgs = np.asarray([io.imread(Path(dir_images, file)) for file in files])

    basic = BaSiC(
        get_darkfield=True, smoothness_flatfield=1, max_reweight_iterations=1000
    )
    basic.fit(imgs)
    imgs_corr = basic.transform(imgs, timelapse=correction_mode)
    imgs_corr = np.where(imgs_corr < 0, 0, imgs_corr)
    imgs_corr = np.where(imgs_corr > 65535, 65535, imgs_corr).astype(np.uint16)
    if median_filter:
        imgs_corr = np.asarray([median(img, morphology.disk(8)) for img in imgs_corr])
    [
        io.imsave(Path(dir_output, file), img, check_contrast=False)
        for file, img in zip(files, imgs_corr)
    ]


def process_plate(
    dir_plate: str, input_type: str, config: dict, df_metadata_plate: pd.DataFrame
) -> None:
    """
    Process plate.
    :param dir_plate:
    :param input_type:
    :param config:
    :param df_metadata:
    :param df_metadata_plate:
    :return:
    """

    dir_output = Path(dir_plate, f"{input_type}_BG")
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    if config["mode"] == "basicpy":
        # get metadata
        df_images = get_metadata(Path(dir_plate, input_type))
        # get unique wells and channel combinations
        df_iter = df_images.groupby(["well_id", "channel_id"]).size().reset_index()
        # new mode column: all additive
        df_iter["mode"] = "additive"
        df_iter["median_filter"] = False
        plate_id = df_metadata_plate["plate_id"].unique().tolist()[0]
        if "multiplicative" in config:
            if plate_id in config["multiplicative"]:
                multiplicative_channels = config["multiplicative"][plate_id]
                # set mode to multiplicative
                df_iter.loc[
                    df_iter["channel_id"].isin(multiplicative_channels), "mode"
                ] = "multiplicative"
        # same for median filter
        if "median_filter" in config:
            if plate_id in config["median_filter"]:
                median_filter_channels = config["median_filter"][plate_id]
                df_iter.loc[
                    df_iter["channel_id"].isin(median_filter_channels), "median_filter"
                ] = True

        # setup dask client
        client, cluster = setup_dask_client(
            memory="64GB",
            n_processes=4,
            n_cores=4,
            gpu=False,
            task_name=f"bleach_correction_{input_type}",
            log_directory=Path(dir_plate, "logs"),
        )
        # bleach correction
        process_well_bleach_correction_partial = partial(
            process_well_bleach_correction,
            dir_plate=dir_plate,
            input_type=input_type,
            dir_output=dir_output,
        )
        results_bleach_corr = []
        futures = client.map(
            process_well_bleach_correction_partial,
            *[
                df_iter["well_id"].to_list(),
                df_iter["channel_id"].to_list(),
                df_iter["median_filter"].tolist(),
                df_iter["mode"].tolist(),
            ],
            pure=False,
        )
        for future, result in as_completed(futures, with_results=True):
            results_bleach_corr.append(result)
        if cluster is not None:
            cluster.close()
        client.close()
    else:
        # get metadata
        df_iter = get_metadata(Path(dir_plate, input_type))
        wells = df_iter["well_id"].unique().tolist()
        # estimate backgrounds
        client, cluster = setup_dask_client(
            memory="32GB",
            n_processes=4,
            n_cores=4,
            gpu=False,
            task_name=f"background_{input_type}",
            log_directory=Path(dir_plate, "logs"),
        )
        dir_output_bg = Path(dir_plate, "background")
        Path(dir_output_bg).mkdir(parents=True, exist_ok=True)
        df_bg = []
        process_well_partial = partial(
            process_well_estimate_bg, dir_plate=dir_plate, input_type=input_type
        )
        futures = client.map(process_well_partial, wells, pure=False)
        for future, result in as_completed(futures, with_results=True):
            df_bg.append(result)
        df_bg = pd.concat(
            [
                pd.read_csv(file, dtype={"channel": object, "z_slice": object})
                for file in df_bg
            ]
        )
        df_bg.to_csv(Path(dir_output_bg, f"df_bg_{input_type}.csv"), index=False)

        # background correction
        correct_background_partial = partial(
            correct_background,
            dir_plate=dir_plate,
            input_type=input_type,
            dir_output=dir_output,
        )
        results_bg = []
        futures = client.map(
            correct_background_partial,
            *[
                df_bg["well"].to_list(),
                df_bg["channel"].to_list(),
                df_bg["z_slice"].to_list(),
                df_bg["bg_value"].to_list(),
            ],
            pure=False,
        )

        for future, result in as_completed(futures, with_results=True):
            results_bg.append(result)

        if cluster is not None:
            cluster.close()
        client.close()

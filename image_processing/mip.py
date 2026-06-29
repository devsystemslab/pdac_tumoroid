from functools import partial
from pathlib import Path

import numpy as np
from dask.distributed import as_completed
from skimage import io

from image_processing.utils import get_metadata, setup_dask_client


def process_images(df_images, dir_output_plate, dir_images):
    """
    Process images for a given well and channel
    :param well:
    :param channel:
    :param dir_output_plate:
    :param dir_images:
    :param df_images:
    :return:
    """
    df_images = df_images.sort_values(by=["field_id", "z_stack_id"], ascending=True)
    # process images
    files = df_images["file"].tolist()
    imgs = [io.imread(Path(dir_images, file)) for file in files]
    imgs = np.asarray(imgs)
    # run z-projection
    img = np.max(imgs, axis=0)
    # save image
    file = Path(dir_output_plate, files[0])
    io.imsave(file, img, check_contrast=False)
    return files


def process_well_mip(well, channel, dir_output_plate, dir_images):
    """
    Process images for a given well and channel
    :param well:
    :param channel:
    :param dir_output_plate:
    :param dir_images:
    :return:
    """
    # get images for well and channel
    df_images = get_metadata(dir_images)
    df_images = df_images[(df_images["well_id"] == well) & (df_images["channel_id"] == channel)]
    # process images
    processed_files = process_images(df_images=df_images, dir_output_plate=dir_output_plate, dir_images=dir_images)
    return processed_files


def process_plate(dir_plate, input_type):
    """
    Process images for a given plate
    :param dir_plate:
    :param input_type:
    :return:
    """
    client, cluster = setup_dask_client(
        memory="16GB",
        walltime="1:00",
        n_processes=2,
        n_cores=2,
        gpu=False,
        task_name=f"mip_{input_type}",
        log_directory=Path(dir_plate, "logs"),
    )
    # get image metadata
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)
    output_type = input_type.replace("TIF_", "TIF_MIP_")
    # set up output directory
    dir_output_plate = Path(dir_plate, output_type)
    dir_output_plate.mkdir(parents=True, exist_ok=True)

    # process images
    processed_files = []
    df_iter = df_images.groupby(["well_id", "channel_id"]).size().reset_index()
    process_well_partial = partial(process_well_mip, dir_output_plate=dir_output_plate, dir_images=dir_images)
    futures = client.map(
        process_well_partial,
        *[df_iter["well_id"].to_list(), df_iter["channel_id"].to_list()],
        pure=False,
    )
    futures = client.map(
        process_well_partial, *[df_iter["well_id"].to_list(), df_iter["channel_id"].to_list()], pure=False
    )
    for future, result in as_completed(futures, with_results=True):
        processed_files.extend(result)
    if cluster is not None:
        cluster.close()
    client.close()

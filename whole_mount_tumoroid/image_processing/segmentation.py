from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from cellpose import denoise, models
from dask.distributed import as_completed
from skimage import io
from skimage.exposure import equalize_adapthist

from whole_mount_tumoroid.image_processing.utils import get_metadata, setup_dask_client


def run_cellpose(
    img: np.ndarray,
    diameter: int = 25,
    model_type: str = "cyto3",
    flow_threshold: float = 0.5,
    cellprob_threshold: float = 0,
    adapt_hist: bool = True,
    scale: bool = False,
    restore_dapi: bool = True,
) -> np.ndarray:
    """
    Run cellpose on an image
    :param img:
    :param diameter:
    :param model_type:
    :param flow_threshold:
    :param cellprob_threshold:
    :param adapt_hist:
    :param scale:
    :param restore_dapi:
    :return:
    """
    if adapt_hist:
        img = equalize_adapthist(img)
    if restore_dapi:
        model = denoise.CellposeDenoiseModel(
            gpu=True,
            model_type=model_type,
            restore_type=f"denoise_{model_type}",
            chan2_restore=True,
        )
        mask, flows, styles, imgs_dn = model.eval(
            img,
            diameter=diameter,
            channels=[[0, 0]],
            normalize=scale,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    else:
        model = models.Cellpose(gpu=True, model_type=model_type)
        mask, flows, styles, diams = model.eval(
            img,
            diameter=diameter,
            channels=[[0, 0]],
            normalize=scale,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    return mask


def process_images(
    dir_output_plate: str, dir_images: str, df_images: pd.DataFrame, restore_dapi: bool
) -> list:
    """
    Process images for a given well and channel
    :param dir_output_plate:
    :param dir_images:
    :param df_images:
    :param restore_dapi:
    :return:
    """
    # load images and run cellpose
    files = df_images["file"].tolist()
    # save images
    for i in range(len(files)):
        img = run_cellpose(
            io.imread(Path(dir_images, files[i])), restore_dapi=restore_dapi
        )
        file = Path(dir_output_plate, files[i])
        io.imsave(file, img, check_contrast=False)
    return files


def process_well_segmentation(
    well: str, dir_output_plate: str, dir_images: str, restore_dapi: bool
) -> list:
    """
    Process images for a given well and channel
    :param well:
    :param dir_output_plate:
    :param dir_images:
    :param restore_dapi:
    :return:
    """
    # get images for well and channel
    df_images = get_metadata(dir_images)
    df_images = df_images[
        (df_images["well_id"] == well) & (df_images["channel_id"] == "01")
    ]
    # process images
    processed_files = process_images(
        df_images=df_images,
        dir_output_plate=dir_output_plate,
        dir_images=dir_images,
        restore_dapi=restore_dapi,
    )
    return processed_files


def process_plate(dir_plate: str, input_type: str, restore_dapi: bool) -> None:
    """
    Process images for a given plate
    :param dir_plate:
    :param input_type:
    :param restore_dapi:
    :return:
    """

    client, cluster = setup_dask_client(
        memory="32GB",
        walltime="4:00",
        n_processes=2,
        n_cores=2,
        n_jobs=20,
        gpu=True,
        task_name=f"segmentation_{input_type}",
        log_directory=Path(dir_plate, "logs"),
    )

    # get image metadata
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)

    # set up output directory
    dir_output_plate = Path(dir_plate, f"SEG_{input_type}")
    dir_output_plate.mkdir(parents=True, exist_ok=True)

    # process images
    processed_files = []
    df_iter = df_images.groupby(["well_id"]).size().reset_index()
    process_well_partial = partial(
        process_well_segmentation,
        dir_output_plate=dir_output_plate,
        dir_images=dir_images,
        restore_dapi=restore_dapi,
    )
    futures = client.map(process_well_partial, df_iter["well_id"].to_list(), pure=False)
    for future, result in as_completed(futures, with_results=True):
        processed_files.extend(result)
    if cluster is not None:
        cluster.close()
    client.close()

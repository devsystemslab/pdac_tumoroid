import os
from pathlib import Path

import numpy as np
from basicpy import BaSiC
from dask.distributed import wait
from skimage import io

from image_processing.utils import get_metadata, setup_dask_client


def calculate_flatfield(files, dir_images, dir_output=None, max_iter=1000):
    """
    Calculate flatfield image from a set of images
    :param files:
    :param dir_images:
    :param dir_output:
    :param max_iter:
    :return:
    """
    images = np.asarray([io.imread(Path(dir_images, file)) for file in files])
    # create a BaSiC object
    basic = BaSiC(
        get_darkfield=False, smoothness_flatfield=1, max_reweight_iterations=max_iter
    )
    # fit the flatfield
    basic.fit(images)

    file_out = Path(dir_output, files[0])
    io.imsave(file_out, basic.flatfield, check_contrast=False)


def run_basic(
    dir_images,
    chunk_size=250,
    dir_plate=None,
    n_samples=None,
    remove_first_tile=False,
    wells=None,
    max_iter=1000,
):
    """
    Calculate flatfields for a plate
    :param dir_images:
    :param chunk_size:
    :param dir_plate:
    :param n_samples:
    :param remove_first_tile:
    :param wells:
    :param max_iter:
    :return:
    """
    client, cluster = setup_dask_client(
        memory="64GB",
        walltime="6:00",
        n_processes=4,
        n_cores=4,
        gpu=False,
        task_name="flatfield",
        log_directory=Path(dir_plate, "logs"),
    )
    dir_output = Path(dir_plate, "flatfield")
    Path(dir_output).mkdir(parents=True, exist_ok=True)
    dir_tmp = Path(dir_output, "tmp")
    dir_tmp.mkdir(parents=True, exist_ok=True)
    df = get_metadata(dir_images)
    if wells is not None:
        df = df[df["well_id"].isin(wells)]
    # remove first tile
    if remove_first_tile:
        df = df[df["field_id"] != "001"]
    channels = df["channel_id"].unique().tolist()
    futures = []
    for channel_id in channels:
        df_images = df[df["channel_id"] == channel_id]

        n_samples_max = int(np.floor(df_images.shape[0] / chunk_size))

        if n_samples is None or n_samples > n_samples_max:
            if n_samples_max == 0:
                n_samples = 1
                chunk_size = df_images.shape[0]
            else:
                n_samples = n_samples_max
        for i in range(n_samples):
            # sample images
            df_images_sampled = df_images.sample(chunk_size)
            # remove sampled images from df_images
            df_images = df_images[~df_images["file"].isin(df_images_sampled["file"])]
            futures.append(
                client.submit(
                    calculate_flatfield,
                    df_images_sampled["file"].values,
                    dir_images,
                    dir_tmp,
                    max_iter,
                )
            )
    try:
        wait(futures, timeout="90 minutes", return_when="ALL_COMPLETED")
    except Exception:
        for f in futures:
            if f.status != "finished":
                f.cancel()

    df_flatfields = get_metadata(dir_tmp)
    for channel_id in channels:
        imgs_flatfield = df_flatfields[df_flatfields["channel_id"] == channel_id][
            "file"
        ].tolist()
        if len(imgs_flatfield) == 0:
            raise ValueError(f"No flatfield images found for channel {channel_id}!")
        if len(imgs_flatfield) == 1:
            img_flatfield_mean = io.imread(Path(dir_tmp, imgs_flatfield[0]))
        else:
            img_flatfield_mean = np.mean(
                np.asarray([io.imread(Path(dir_tmp, file)) for file in imgs_flatfield]),
                axis=0,
            )
        # save
        file_out = Path(dir_output, f"flatfield_ch{channel_id}.tif")
        io.imsave(file_out, img_flatfield_mean)

    # del tmp folder
    os.system(f"rm -r {dir_tmp}")
    # close client and cluster
    if cluster is not None:
        cluster.close()
    client.close()

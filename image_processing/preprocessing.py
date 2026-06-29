from functools import partial
from pathlib import Path

import numpy as np
from dask.distributed import as_completed
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import montage

from image_processing.utils import get_metadata, setup_dask_client


def load_flatfield(channel, dir_flatfield):
    """
    Load flatfield image for a given plate and channel
    :param plate_id: string of plate id
    :param channel: string of channel id
    :param dir_flatfield: path to flatfield image directory
    :return: flatfield image
    """
    file = f"flatfield_ch{channel}.tif"
    if Path(dir_flatfield, file).exists():
        img = io.imread(Path(dir_flatfield, file))
        return img
    else:
        return None


def correct_image(img, img_flatfield):
    """
    Correct image for illumination and background
    :param img: image to correct
    :param img_flatfield: flatfield image
    :return: corrected image
    """
    if img_flatfield is None:
        return img
    else:
        img_corrected = img / img_flatfield
        img_corrected = img_corrected.astype(np.uint16)
        return img_corrected


def denoise_image(img):
    """
    Denoise image
    :param img:
    :return:
    """
    sigma_est = np.mean(estimate_sigma(img))

    patch_kw = dict(patch_size=5, patch_distance=6)

    img = denoise_nl_means(
        img,
        h=0.8 * sigma_est,
        sigma=sigma_est,
        preserve_range=True,
        fast_mode=True,
        **patch_kw,
    )
    img = img.astype(np.uint16)
    return img


def stitch_images(imgs, img_diff_dim=None, ref_overlap=0, overlap=0.1, axis=0):
    """
    Stitch images together
    :param imgs: images to stitch
    :param img_diff_dim:
    :param ref_overlap:
    :type overlap: float
    :param axis: axis to stitch along
    :return: stitched image
    """
    img_ref = None
    img_add = None

    if len(imgs.shape) not in [2, 3]:
        raise ValueError("imgs must be 2x2 or 2x3 array")

    if len(imgs.shape) == 3:
        img_ref = imgs[0]
        img_add = imgs[1]
    if img_diff_dim is not None and len(imgs.shape) == 2:
        img_ref = imgs
        img_add = img_diff_dim
    if img_ref is None or img_add is None:
        raise ValueError("imgs must be 2x2 or 2x3 array")
    # get image shape reference image
    if ref_overlap == 0:
        shape = img_ref.shape
    elif ref_overlap == 1:
        shape = img_add.shape
    else:
        raise ValueError("ref_overlap must be 0 or 1")
    # get overlap
    overlap = int(shape[axis] * overlap)
    # get linear blending for overlap array
    blend = np.linspace(1, 0, overlap)
    if axis == 0:
        # make 2D array from blend
        blend = np.tile(blend, (shape[1 - axis], 1)).T
        overlap_img = (img_ref[-overlap:, :] * blend) + (
            img_add[:overlap, :] * (1 - blend)
        )
        img = np.concatenate(
            [img_ref[:-overlap, :], overlap_img, img_add[overlap:, :]], axis=axis
        )
    elif axis == 1:
        # make 2D array from blend
        blend = np.tile(blend, (shape[1 - axis], 1))
        overlap_img = (img_ref[:, -overlap:] * blend) + (
            img_add[:, :overlap] * (1 - blend)
        )
        img = np.concatenate(
            [img_ref[:, :-overlap], overlap_img, img_add[:, overlap:]], axis=axis
        )
    else:
        raise ValueError("axis must be 0 or 1")
    return img


def stitch_montage(imgs, overlap, correction_factor=0.007):
    overlap = overlap - correction_factor
    n_images = imgs.shape[0]
    if n_images == 4:
        # stitch vertically
        imgs = np.asarray(
            [
                stitch_images(imgs[::2], overlap=overlap, axis=0),
                stitch_images(imgs[1::2], overlap=overlap, axis=0),
            ]
        )
        # stitch horizontally
        img = stitch_images(imgs, overlap=overlap, axis=1)
    elif n_images == 9:
        # stitch first row
        imgs_first_row = stitch_images(
            stitch_images(imgs[0], imgs[1], overlap=overlap, axis=1),
            imgs[2],
            overlap=overlap,
            axis=1,
        )
        # stitch second row
        imgs_second_row = stitch_images(
            stitch_images(imgs[3], imgs[4], overlap=overlap, axis=1),
            imgs[5],
            overlap=overlap,
            axis=1,
        )
        # stitch third row
        imgs_third_row = stitch_images(
            stitch_images(imgs[6], imgs[7], overlap=overlap, axis=1),
            imgs[8],
            overlap=overlap,
            axis=1,
        )
        # stitch vertically
        imgs = np.asarray([imgs_first_row, imgs_second_row, imgs_third_row])
        img = stitch_images(imgs[0], imgs[1], overlap=overlap, axis=0)
        img = stitch_images(img, imgs[2], overlap=overlap, axis=0)
    elif n_images == 16:
        # stitch first row
        imgs_first_row = stitch_images(
            stitch_images(imgs[0], imgs[1], overlap=overlap, axis=1),
            imgs[2],
            overlap=overlap,
            axis=1,
        )
        # stitch second row
        imgs_second_row = stitch_images(
            stitch_images(imgs[3], imgs[4], overlap=overlap, axis=1),
            imgs[5],
            overlap=overlap,
            axis=1,
        )
        # stitch third row
        imgs_third_row = stitch_images(
            stitch_images(imgs[6], imgs[7], overlap=overlap, axis=1),
            imgs[8],
            overlap=overlap,
            axis=1,
        )
        # stitch fourth row
        imgs_fourth_row = stitch_images(
            stitch_images(imgs[9], imgs[10], overlap=overlap, axis=1),
            imgs[11],
            overlap=overlap,
            axis=1,
        )
        # stitch vertically
        imgs = np.asarray(
            [imgs_first_row, imgs_second_row, imgs_third_row, imgs_fourth_row]
        )
        img = stitch_images(imgs[0], imgs[1], overlap=overlap, axis=0)
        img = stitch_images(img, imgs[2], overlap=overlap, axis=0)
        img = stitch_images(img, imgs[3], overlap=overlap, axis=0)
    else:
        raise ValueError("Number of images not 4, 9 or 16.")
    return img


def process_images(
    well,
    z_plane,
    channel,
    dir_output_plate,
    img_flatfield,
    dir_images,
    df_images,
    overlap=None,
):

    df_images_select = df_images[
        (df_images["well_id"] == well)
        & (df_images["channel_id"] == channel)
        & (df_images["z_stack_id"] == z_plane)
    ]
    # arrange df_images_select by position
    df_images_select = df_images_select.sort_values(by=["field_id"], ascending=True)
    # load images
    files = df_images_select["file"].tolist()
    if len(files) not in [4, 9, 16]:
        print(
            f"Found {len(files)} images for well {well}, z_plane {z_plane}, channel {channel}. Which is not 4, 9 or 16."
        )
        return []
    imgs = [
        correct_image(io.imread(Path(dir_images, file)), img_flatfield)
        for file in files
    ]
    imgs = np.asarray(imgs)
    # run montage
    if overlap is not None:
        img = stitch_montage(imgs, overlap)
    else:
        img = montage(imgs)
    # denoise image
    img = denoise_image(img)
    # save image
    file = Path(dir_output_plate, files[0])
    io.imsave(file, img, check_contrast=False)
    return files


def process_well_preprocessing(
    well,
    channel,
    dir_output,
    dir_output_plate,
    dir_images,
    overlap=None,
    correct=True,
    remove_first_tile=True,
):

    # load flatfield image
    if correct:
        img_flatfield = load_flatfield(
            channel, dir_flatfield=Path(dir_output, "flatfield")
        )
    else:
        img_flatfield = None
    # get images for well and channel
    df_images = get_metadata(dir_images)
    df_images = df_images[(df_images["channel_id"] == channel)]
    df_images = df_images[df_images["well_id"] == well]
    # filter field 1 if necessary
    if remove_first_tile:
        df_images = df_images[df_images["field_id"] != "001"]
    # z-plane and get unique combinations
    df_iter = df_images.groupby(["z_stack_id"]).size().reset_index()
    processed_files = []
    # run process_images
    for z_plane in df_iter["z_stack_id"].tolist():
        processed_files.extend(
            process_images(
                well=well,
                z_plane=z_plane,
                channel=channel,
                dir_output_plate=dir_output_plate,
                img_flatfield=img_flatfield,
                dir_images=dir_images,
                df_images=df_images,
                overlap=overlap,
            )
        )

    return processed_files


def process_plate(df_plate, dir_plate, remove_first_tile=False, wells=None):
    """
    Process images for a given plate
    :param df_plate: dataframe with plate metadata
    :param dir_plate: path to plate directory
    :param remove_first_tile: whether to remove the first tile
    :param wells: list of wells to process, if None process all wells
    :return:
    """
    client, cluster = setup_dask_client(
        walltime="6:00",
        n_processes=10,
        n_cores=10,
        gpu=False,
        task_name="preprocessing",
        log_directory=Path(dir_plate, "logs"),
    )
    # get image metadata
    if "tile_overlap" not in df_plate.columns:
        overlap = None
    else:
        overlap = df_plate["tile_overlap"].iloc[0]
    dir_images = df_plate["dir_raw"].iloc[0]
    df_images = get_metadata(dir_images)

    # set up output directory
    dir_output_plate = Path(dir_plate, "TIF_OVR")
    dir_output_plate.mkdir(parents=True, exist_ok=True)

    # process images
    processed_files = []
    df_iter = df_images.groupby(["well_id", "channel_id"]).size().reset_index()
    if wells is not None:
        df_iter = df_iter[df_iter["well_id"].isin(wells)]
    process_well_partial = partial(
        process_well_preprocessing,
        overlap=overlap,
        dir_output=dir_plate,
        dir_output_plate=dir_output_plate,
        dir_images=dir_images,
        correct=True,
        remove_first_tile=remove_first_tile,
    )
    futures = client.map(
        process_well_partial,
        *[df_iter["well_id"].to_list(), df_iter["channel_id"].to_list()],
        pure=False,
    )
    for future, result in as_completed(futures, with_results=True):
        processed_files.extend(result)
    if cluster is not None:
        cluster.close()
    client.close()

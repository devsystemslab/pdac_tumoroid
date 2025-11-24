import os
from functools import partial
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from dask.distributed import as_completed
from laminator import LaminarNeighbors
from skimage import io

from whole_mount_tumoroid.image_processing.utils import get_metadata, setup_dask_client

# set environment variable TQDM_DISABLE=1
# to disable tqdm progress bar
os.environ["TQDM_DISABLE"] = "1"


def regular_grid_neighbors(sample, z, dir_plate, input_type, dir_out, dir_tmp):
    df_images = get_metadata(Path(dir_plate, input_type))
    # filter for well and z
    df_images = df_images[
        (df_images["well_id"] == sample) & (df_images["z_stack_id"] == z)
    ]
    # sort by channel_id
    df_images = df_images.sort_values(by="channel_id")
    # run laminator for feature extraction
    la = LaminarNeighbors(radius=50)
    img_files = [
        Path(dir_images, file)
        for dir_images, file in zip(df_images["dir_images"], df_images["file"])
    ]
    la.load_images(img_files)
    la.set_image_names(
        [f"ch_{channel}" for channel in df_images["channel_id"].to_list()]
    )
    # load open mask
    la.mask = np.ones(la.images[..., 0].shape)
    positions_x = np.arange(50, la.images.shape[1] + 1, 50)
    positions_y = np.arange(50, la.images.shape[0] + 1, 50)
    idx_x = np.arange(0, len(positions_x))
    idx_y = np.arange(0, len(positions_y))
    # generate rasterscan positions
    positions = np.array(np.meshgrid(positions_x, positions_y)).T.reshape(-1, 2)
    idx_positions = np.array(np.meshgrid(idx_x, idx_y)).T.reshape(-1, 2)
    df_positions = pd.DataFrame(positions, columns=["x", "y"])
    df_positions["idx_x"] = idx_positions[:, 0]
    df_positions["idx_y"] = idx_positions[:, 1]
    # add label to each position by setting index as column label
    df_positions["label"] = df_positions.index
    # add to laminator object
    la.positions = df_positions
    la.calculate_radial_profiles()
    la.average_intensities()
    df_merged = la.positions.merge(la.avg_intensities, on=["label", "x", "y"])
    df_pivot = df_merged.pivot(
        index=["idx_x", "idx_y"], columns="image", values="intensity"
    )
    # use idx_x and idx_y as index for creating a 2D array
    imgs = np.zeros((len(idx_y), len(idx_x), len(df_pivot.columns)))
    imgs[
        df_pivot.index.get_level_values("idx_y"),
        df_pivot.index.get_level_values("idx_x"),
        :,
    ] = df_pivot.values
    file = Path(dir_out, f"{sample}_{z}.tif")
    io.imsave(file, imgs.astype(np.uint16), check_contrast=False)
    df = pd.DataFrame({"well_id": [sample], "z_stack_id": [z], "file": [file]})
    file_df = Path(dir_tmp, f"{sample}_{z}.csv")
    df.to_csv(file_df, index=False)
    return file_df


def message_passing(sample, z, dir_plate, input_type, segmentation_input, dir_out):
    df_images = get_metadata(Path(dir_plate, input_type))
    df_label = get_metadata(Path(dir_plate, segmentation_input))
    df_label = df_label[(df_label["well_id"] == sample) & (df_label["z_stack_id"] == z)]
    if df_label.shape[0] == 0:
        return sample
    df_select = df_images[
        (df_images["well_id"] == sample) & (df_images["z_stack_id"] == z)
    ]
    # sort by channel_id
    df_select = df_select.sort_values(by="channel_id")
    # load positions
    la = LaminarNeighbors(radius=50)
    la.load_label_image(
        path=Path(dir_plate, segmentation_input, df_label["file"].iloc[0])
    )
    # get number of labels
    n_labels = np.unique(la.label_image).shape[0]
    if n_labels <= 2:
        return sample
    else:
        # load images
        la.load_images(
            [Path(dir_plate, input_type, file) for file in df_select["file"]]
        )
        la.set_image_names(
            [f"ch_{channel}" for channel in df_select["channel_id"].to_list()]
        )
        # load open mask
        la.mask = np.ones(la.label_image.shape)
        # run laminator
        la.extract_positions()
        la.calculate_radial_profiles()
        la.average_intensities()
        la.louvain_labels = np.ones(la.positions.shape[0])
        la.build_graph(n_neigh=3, radius=100)
        la.get_angles()
        la.get_path_intensities()
        la.construct_features(scale=False)
        # save 'message_passed_intensities' to adata and ad la.positions as obs
        adata = ad.AnnData(la.msg_passed_intensities)
        # set image_names as feature names
        adata.var_names = list(la.image_names.values())
        adata.obs = la.positions
        adata.obs_names = la.positions.index.astype(str)
        # add sample
        adata.obs["sample"] = sample
        # all to one dataframe
        df_features = pd.concat([adata.obs, adata.to_df()], axis=1)
        df_features.to_csv(Path(dir_out, f"{sample}_{z}.csv"))
        return sample


def process_plate(dir_plate, input_type, segmentation_input):
    """
    Process plate.
    :param dir_plate:
    :param input_type:
    :param segmentation_input:
    :return:
    """
    client, cluster = setup_dask_client(
        memory="32GB",
        walltime="4:00",
        n_processes=4,
        n_cores=4,
        n_jobs=40,
        task_name=f"message_passing_{input_type}",
        log_directory=Path(dir_plate, "logs"),
    )
    # set up output directory
    dir_out = Path(dir_plate, "features", "laminator", input_type, "neighbors")
    Path.mkdir(dir_out, parents=True, exist_ok=True)
    # load metadata
    df_images = get_metadata(Path(dir_plate, input_type))
    # get wells
    df_iter = df_images.groupby(["well_id", "z_stack_id"]).size().reset_index()
    # message passing
    message_passing_partial = partial(
        message_passing,
        dir_plate=dir_plate,
        input_type=input_type,
        segmentation_input=segmentation_input,
        dir_out=dir_out,
    )
    futures = client.map(
        message_passing_partial,
        *[df_iter["well_id"].to_list(), df_iter["z_stack_id"].to_list()],
        pure=False,
    )
    processed_files = []
    for future, result in as_completed(futures, with_results=True):
        processed_files.extend(result)

    if cluster is not None:
        cluster.close()
    client.close()

    # regular grid neighbors
    client, cluster = setup_dask_client(
        memory="16GB",
        walltime="2:00",
        n_processes=2,
        n_cores=2,
        n_jobs=40,
        task_name=f"regular_grid_{input_type}",
        log_directory=Path(dir_plate, "logs"),
    )
    dir_out = Path(dir_plate, "features", "laminator", input_type, "regular_grid")
    Path.mkdir(dir_out, parents=True, exist_ok=True)
    dir_tmp = Path(dir_out, "tmp")
    Path.mkdir(dir_tmp, parents=True, exist_ok=True)
    df_summary = []
    regular_grid_neighbors_partial = partial(
        regular_grid_neighbors,
        dir_plate=dir_plate,
        input_type=input_type,
        dir_out=dir_out,
        dir_tmp=dir_tmp,
    )
    futures = client.map(
        regular_grid_neighbors_partial,
        *[df_iter["well_id"].to_list(), df_iter["z_stack_id"].to_list()],
        pure=False,
    )
    for future, result in as_completed(futures, with_results=True):
        df_summary.append(result)
    df_summary = pd.concat([pd.read_csv(file) for file in df_summary], axis=0)
    df_summary.to_csv(
        Path(dir_plate, "features", "laminator", input_type, "df_regular_grid.csv"),
        index=False,
    )
    # delete temporary files
    os.system(f"rm -r {dir_tmp}")
    # close client
    if cluster is not None:
        cluster.close()
    client.close()

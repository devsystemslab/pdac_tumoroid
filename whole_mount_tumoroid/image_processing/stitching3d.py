from whole_mount_tumoroid.image_processing.utils import setup_dask_client, get_metadata
from pathlib import Path
from skimage import io
import numpy as np
from cellpose.utils import stitch3D
from functools import partial
from dask.distributed import as_completed

def merge_labels(label_img):
    # unique labels across all z-slices
    for i in range(1,label_img.shape[0]):
        label_img[i] = np.where(label_img[i] > 0,label_img[i] + label_img[i-1].max(), label_img[i])
    label_img = stitch3D(label_img)
    return label_img

def process_well_merge_labels(well, dir_images):
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
    df_images = df_images[(df_images['well_id'] == well)]
    # sort by z_stack
    df_images = df_images.sort_values(by='z_stack_id', ascending=True).reset_index(drop=True)
    # load labels
    label_img = np.asarray([io.imread(Path(row['dir_images'],row['file'])) for index, row in df_images.iterrows()])
    label_img = merge_labels(label_img)
    # save images
    [io.imsave(Path(dir_images, row['dir_images'], row['file']),label_img[index]) for index, row in df_images.iterrows()]
    return well

def process_plate(dir_plate, input_type):
    client, cluster = setup_dask_client(memory='150GB', walltime='12:00', n_processes=1, n_cores=1, n_jobs=80,
                                        task_name=f'merge_labels_{input_type}', log_directory=Path(dir_plate, 'logs'))
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)
    # process images
    processed_files = []
    df_iter = df_images.groupby(['well_id']).size().reset_index()
    process_well_partial = partial(process_well_merge_labels, dir_images=dir_images)
    futures = client.map(process_well_partial, df_iter['well_id'].to_list(), pure=False)
    for future, result in as_completed(futures, with_results=True):
        processed_files.extend(result)
    if cluster is not None:
        cluster.close()
    client.close()

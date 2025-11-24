from whole_mount_tumoroid.image_processing.utils import setup_dask_client, get_metadata
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np
from skimage import io
from skimage.filters import threshold_otsu
from skimage.measure import regionprops_table
from dask.distributed import as_completed

def detect_empty_images(imgs):
    return any([len(imgs) == 0 for imgs in imgs])

def foreground_ratio(imgs, channel=0):
    # TODO: consider not maxing over all channels but using the channel with the highest foreground ratio
    threshold = threshold_otsu(np.max(imgs, axis=0))
    ratio = np.sum(np.max(imgs, axis=0) > threshold) / imgs[channel, ...].size
    return ratio

def test_foreground_ratio(imgs, channel=0, ratio_threshold=0.005):
    ratio = foreground_ratio(imgs, channel)
    if ratio > ratio_threshold:
        return True
    else:
        return False

def process_well_features(well, z_stack, dir_images, dir_segmented, dir_output_plate):

    # load metadata
    df_images = get_metadata(dir_images)
    df_images = df_images[(df_images['well_id'] == well) &
                          (df_images['z_stack_id'] == z_stack)]
    # arrange by imaging cycle and channel
    df_images = df_images.sort_values(by='channel_id')

    # get metadata for label images
    df_label_images = get_metadata(dir_segmented)
    df_label_images = df_label_images[(df_label_images['z_stack_id'] == z_stack) &
                                      (df_label_images['well_id'] == well)]

    if df_label_images.shape[0] == 0:
        return []

    assert df_label_images.shape[0] == 1

    # load images
    img_label = io.imread(Path(df_label_images['dir_images'].iloc[0], df_label_images['file'].iloc[0]))

    imgs = np.asarray([io.imread(Path(df_images['dir_images'].iloc[i], df_images['file'].iloc[i])) for i in
                       range(df_images.shape[0])])

    if detect_empty_images(imgs):
        return []

    if not test_foreground_ratio(imgs):
        return []

    df = pd.DataFrame(regionprops_table(label_image=img_label,
                                        intensity_image=np.moveaxis(imgs, 0, -1),
                                        properties=('label', 'centroid', 'area','eccentricity',
                                                    'intensity_mean', 'major_axis_length', 'minor_axis_length')))
    # add metadata
    df['well'] = well
    df['z_stack'] = z_stack

    # write to csv
    df.to_csv(Path(dir_output_plate, f'{well}_{z_stack}.csv'), index=False)

    # per image summary statistics
    df_summary = pd.DataFrame()
    df_summary['dir_images'] = df_images['dir_images']
    df_summary['file'] = df_images['file']
    df_summary['well'] = well
    df_summary['z_stack'] = z_stack
    df_summary['channel'] = df_images['channel_id']
    df_summary['foreground_ratio'] = np.asarray([foreground_ratio(imgs, channel) for channel in range(imgs.shape[0])])
    df_summary['n_cells'] = df.shape[0]
    df_summary['max'] = np.max(imgs, axis=(1,2))
    df_summary['mean'] = np.mean(imgs, axis=(1,2))
    df_summary['std'] = np.std(imgs, axis=(1,2))
    df_summary['mean_std'] = df_summary['mean'] / df_summary['std']
    df_summary['min'] = np.min(imgs, axis=(1,2))
    df_summary['median'] = np.median(imgs, axis=(1,2))
    df_summary['mad'] = np.median(np.abs(imgs - np.median(imgs, axis=(1,2))[:, None, None]), axis=(1,2))
    df_summary['median_mad'] = df_summary['median'] / df_summary['mad']
    return df_summary


def process_plate(dir_plate, input_type, segmentation_input):
    client, cluster = setup_dask_client(memory='12GB', walltime='6:00', n_processes=2, n_cores=2, gpu=False,
                                        task_name=f'features_{input_type}', log_directory=Path(dir_plate, 'logs'))
    # load metadata
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)
    dir_segmented = Path(dir_plate, segmentation_input)
    # set up output directory
    dir_output_plate = Path(dir_plate, 'features','nuclei', input_type)
    dir_output_plate.mkdir(parents=True, exist_ok=True)

    # process images
    df_summary = []
    df_iter = df_images.groupby(['well_id','z_stack_id']).size().reset_index()
    process_well_partial = partial(process_well_features,
                                   dir_images=dir_images,
                                   dir_segmented=dir_segmented,
                                   dir_output_plate=dir_output_plate)
    futures = client.map(process_well_partial, *[df_iter['well_id'].to_list(),
                                                 df_iter['z_stack_id'].to_list()], pure=False)
    for future, result in as_completed(futures, with_results=True):
        df_summary.append(result)
    df_summary = [df for df in df_summary if len(df) > 0]
    df_summary = pd.concat(df_summary, axis=0, ignore_index=True)
    dir_summary = Path(dir_plate, 'features','nuclei')
    dir_summary.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(Path(dir_summary, f'df_summary_{input_type}.csv'), index=False)
    if cluster is not None:
        cluster.close()
    client.close()

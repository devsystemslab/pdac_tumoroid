import open3d as o3d
import pandas as pd
from whole_mount_tumoroid.image_processing.utils import setup_dask_client, get_metadata
from pathlib import Path
from skimage import io
from functools import partial
from dask.distributed import as_completed
import matplotlib.pyplot as plt
import copy
from sklearn.neighbors import NearestNeighbors
import numpy as np
import os


def preprocess_point_cloud(pcd, voxel_size):
    """
    Preprocess point cloud
    :param pcd:
    :param voxel_size:
    :return:
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
                                                               o3d.geometry.KDTreeSearchParamHybrid(
                                                                   radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def register_pointclouds(source: np.ndarray, target: np.ndarray, voxel_size: int = 50, threshold: int = 50,
                         global_registration: bool = True):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)
    if global_registration:
        source_down, source_fpfh = preprocess_point_cloud(pcd_source, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(pcd_target, voxel_size)

        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)

        registered = o3d.pipelines.registration.registration_icp(pcd_target, pcd_source, threshold,
                                                                 result_ransac.transformation)
    else:
        registered = o3d.pipelines.registration.registration_icp(pcd_target, pcd_source, threshold)
    pcd_registered = copy.deepcopy(pcd_source).transform(registered.transformation)
    # TODO: return rmsd values for qc plotting
    return np.asarray(pcd_registered.points)


def link_points(source: np.ndarray, target: np.ndarray,
                source_label: np.ndarray, target_label: np.ndarray, max_distance: int = 50):
    """
    Link points from source to target
    :param source:
    :param target:
    :return:
    """
    if source.shape[0] < target.shape[0]:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(target)
        distances, indices = nbrs.kneighbors(source)
        selected = distances < max_distance
        source, target = source[np.squeeze(selected), :], target[(indices[selected]), :]
        source_label, target_label = source_label[np.squeeze(selected)], target_label[(indices[selected])]
    else:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(source)
        distances, indices = nbrs.kneighbors(target)
        selected = distances < max_distance
        target, source = target[np.squeeze(selected), :], source[(indices[selected]), :]
        target_label, source_label = target_label[np.squeeze(selected)], source_label[(indices[selected])]
    return source, target, source_label, target_label, distances[selected]


def plot_registration(source, target, moved):
    """
    Plot registration results
    :param xyz_1:
    :param xyz_2:
    :param xyz_3:
    :return:
    """
    fig, ax = plt.subplots(3, 2, figsize=(15, 18))
    # top view
    ax[0, 0].scatter(source[:, 2], source[:, 1], s=0.5, c='red', alpha=0.5)
    ax[0, 0].scatter(target[:, 2], target[:, 1], s=0.5, c='blue', alpha=0.5)
    ax[0, 0].set_aspect('equal', adjustable='box')
    ax[0, 0].title.set_text('Before registration - top view')
    ax[0, 1].scatter(moved[:, 2], moved[:, 1], s=0.5, c='red', alpha=0.5)
    ax[0, 1].scatter(target[:, 2], target[:, 1], s=0.5, c='blue', alpha=0.5)
    ax[0, 1].set_aspect('equal', adjustable='box')
    ax[0, 1].title.set_text('After registration - top view')
    # side view 1
    ax[1, 0].scatter(source[:, 1], source[:, 0], s=0.5, c='red', alpha=0.5)
    ax[1, 0].scatter(target[:, 1], target[:, 0], s=0.5, c='blue', alpha=0.5)
    ax[1, 0].set_aspect('equal', adjustable='box')
    ax[1, 0].title.set_text('Before registration - side view 1')
    ax[1, 1].scatter(moved[:, 1], moved[:, 0], s=0.5, c='red', alpha=0.5)
    ax[1, 1].scatter(target[:, 1], target[:, 0], s=0.5, c='blue', alpha=0.5)
    ax[1, 1].set_aspect('equal', adjustable='box')
    ax[1, 1].title.set_text('After registration - side view 1')
    # side view 2
    ax[2, 0].scatter(source[:, 2], source[:, 0], s=0.5, c='red', alpha=0.5)
    ax[2, 0].scatter(target[:, 2], target[:, 0], s=0.5, c='blue', alpha=0.5)
    ax[2, 0].set_aspect('equal', adjustable='box')
    ax[2, 0].title.set_text('Before registration - side view 2')
    ax[2, 1].scatter(moved[:, 2], moved[:, 0], s=0.5, c='red', alpha=0.5)
    ax[2, 1].scatter(target[:, 2], target[:, 0], s=0.5, c='blue', alpha=0.5)
    ax[2, 1].set_aspect('equal', adjustable='box')
    ax[2, 1].title.set_text('After registration - side view 2')
    plt.tight_layout()
    plt.show()


def plot_registration_3d(source_init, target_init, source_final, target_final):
    """
    Plot registration results
    :param xyz_1:
    :param xyz_2:
    :param xyz_3:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(source_init[:, 0], source_init[:, 1], source_init[:, 2], s=0.5, alpha=0.5)
    ax.scatter(target_init[:, 0], target_init[:, 1], target_init[:, 2], s=0.5, alpha=0.5)
    ax.title.set_text('Before registration')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(source_final[:, 0], source_final[:, 1], source_final[:, 2], s=0.5, alpha=0.5)
    ax.scatter(target_final[:, 0], target_final[:, 1], target_final[:, 2], s=0.5, alpha=0.5)
    ax.title.set_text('After registration')
    plt.show()


def load_features(well: str, dir_plate: str, cycle: str, input_type: str) -> pd.DataFrame:
    """
    Load features for a given well
    :param well:
    :param dir_plate:
    :param cycle:
    :param input_type:
    :return:
    """
    dir_features = Path(dir_plate, cycle, 'features')
    dir_laminator = Path(dir_features, 'laminator', input_type, 'neighbors')
    dir_nuclei = Path(dir_features, 'nuclei', input_type)
    files = [f for f in os.listdir(dir_laminator) if f.startswith(f'{well}_')]
    if len(files) == 0:
        return None
    df_laminator = pd.concat(
        [pd.read_csv(Path(dir_laminator, f)).assign(z=f.replace(f'{well}_', '').replace('.csv', '')) for f in files])
    # drop columns starting with Unnamed
    df_laminator = df_laminator.loc[:, ~df_laminator.columns.str.contains('^Unnamed')]
    files = [f for f in os.listdir(dir_nuclei) if f.startswith(f'{well}_')]
    if len(files) == 0:
        return None
    df_nuclei = pd.concat(
        [pd.read_csv(Path(dir_nuclei, f)).assign(z=f.replace(f'{well}_', '').replace('.csv', '')) for f in files])
    df = pd.merge(df_laminator, df_nuclei, on=['label', 'z'], how='inner').set_index('label')
    df['z'] = df['z'].astype(int) - 1
    # remove y, x, sample, well, z_stack columns
    columns_remove = ['y', 'x', 'sample', 'well', 'z_stack']
    df = df.drop(columns=columns_remove)
    # add neighbors suffix to all columns that start with ch_01
    columns = df.columns[df.columns.str.contains('^ch_0')]
    for column in columns:
        df.rename(columns={column: f'{column}_neighbors'}, inplace=True)
    # rename intensity_mean-0 to ch_01_intensity_mean
    columns = df.columns[df.columns.str.contains('intensity_mean')]
    # log1p columns
    df[columns] = np.log1p(df[columns])
    for column in columns:
        new_column = int(column.replace('intensity_mean-', '')) + 1
        df.rename(columns={column: f'ch_0{new_column}_nuclei'}, inplace=True)
    if df.shape[0] == 0:
        return None
    else:
        return df


def get_centroids(df: pd.DataFrame, z_step: int, pixel_size: float, filter_area: int = None) -> pd.DataFrame:
    """
    Get centroids from label image
    :param df:
    :param z_step:
    :param pixel_size:
    :param filter_area: int
    :return:
    """
    if df is None:
        return None
    else:
        df['z'] = df['z'] / pixel_size * z_step
        df = df.groupby('label').mean()
        if filter_area is not None:
            df = df[df['area'] > filter_area]
        return df


def process_well_register_nuclei(well: str,
                                 dir_plate: str,
                                 dir_output_plate: str,
                                 source_cycle: str,
                                 target_cycle: str,
                                 input_type: str,
                                 pixel_size: float = 0.322,
                                 z_step: int = 10,
                                 reiterate_after_matching: bool = True,
                                 plot: bool = False) -> str:
    """
    Process images for a given well and channel
    :param well:
    :param dir_plate:
    :param dir_output_plate:
    :param source_cycle:
    :param target_cycle:
    :param input_type:
    :param pixel_size:
    :param z_step:
    :param reiterate_after_matching:
    :param plot:
    :return:
    """
    df_source = get_centroids(load_features(well, dir_plate, source_cycle, input_type), z_step, pixel_size)
    df_target = get_centroids(load_features(well, dir_plate, target_cycle, input_type), z_step, pixel_size)
    if df_source is None or df_target is None:
        return well
    else:
        coordinate_cols = ['z', 'centroid-0', 'centroid-1']
        # select columns and convert to numpy arrays
        source = df_source[coordinate_cols].to_numpy()
        target = df_target[coordinate_cols].to_numpy()
        source_init = source.copy()
        target_init = target.copy()
        source_label = df_source.index.to_numpy()
        target_label = df_target.index.to_numpy()
        # initial registration
        source = register_pointclouds(source, target, voxel_size=100, threshold=50)
        source, target, source_label, target_label, distances = link_points(source, target, source_label, target_label,
                                                                            max_distance=100)
        if source.shape[0] == 0 or target.shape[0] == 0:
            return well
        if reiterate_after_matching:
            # reiterate with match nuclei and smaller voxel size and thresholds
            source = register_pointclouds(source, target, voxel_size=50, threshold=10)
        if plot:
            plot_registration(df_source[coordinate_cols].to_numpy(), target, source)

        # merge df_source and df_target using source_labels and target_labels
        df_source = df_source.loc[source_label].reset_index()
        df_target = df_target.loc[target_label].reset_index()

        # add target and source prefixes to aal columns
        df_source = df_source.add_prefix('source_')
        df_target = df_target.add_prefix('target_')
        df = pd.concat([df_source, df_target], axis=1)
        # add distances
        df['distance'] = distances
        # get unique rows
        df = df.drop_duplicates()
        # set index column as new label column
        df['label'] = df.index
        # move label column to first position
        cols = df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        df = df[cols]
        if plot:
            plot_registration_3d(source_init, target_init, source, target)
        # save to csv
        df.to_csv(Path(dir_output_plate, f'{well}_registration.csv'), index=False)
        return well


def process_plate(dir_plate: str,
                  source_cycle: str,
                  target_cycle: str,
                  input_type: str,
                  z_step: int,
                  pixel_size: float) -> None:
    """
    Process images for a given plate
    :param dir_plate:
    :param source_cycle:
    :param target_cycle:
    :param input_type:
    :param z_step:
    :param pixel_size:
    :return:
    """
    # set up dask client
    client, cluster = setup_dask_client(memory='32GB', walltime='12:00',
                                        n_processes=1, n_cores=3, n_jobs=36,
                                        log_directory=Path(dir_plate, 'logs'),
                                        task_name=f'registration_{input_type}')

    # get plate_information
    df_plates = pd.read_csv(Path(dir_plate, 'plate_information.csv'), dtype={'plate': str})
    dir_images_source = Path(df_plates[df_plates['plate_id'] == source_cycle]['dir_processed'].values[0], input_type)
    dir_images_target = Path(df_plates[df_plates['plate_id'] == target_cycle]['dir_processed'].values[0], input_type)

    # assert both directories exist
    assert dir_images_source.exists()
    assert dir_images_target.exists()

    # get image metadata
    df_images_source = get_metadata(dir_images_source)
    df_images_target = get_metadata(dir_images_target)

    # set up output directory
    dir_output_plate = Path(dir_plate, 'features_registration', input_type)
    dir_output_plate.mkdir(parents=True, exist_ok=True)

    # process images for wells that have both source and target images
    processed_files = []
    df_iter = pd.merge(df_images_source.groupby('well_id').size().reset_index(),
                       df_images_target.groupby('well_id').size().reset_index(), on='well_id', how='inner')
    process_well_partial = partial(process_well_register_nuclei, dir_plate=dir_plate,
                                   dir_output_plate=dir_output_plate,
                                   source_cycle=source_cycle,
                                   target_cycle=target_cycle,
                                   input_type=input_type,
                                   pixel_size=pixel_size,
                                   z_step=z_step,
                                   reiterate_after_matching=True,
                                   plot=False)
    futures = client.map(process_well_partial, df_iter['well_id'].to_list(), pure=False)
    for future, result in as_completed(futures, with_results=True):
        processed_files.extend(result)

    if cluster is not None:
        cluster.close()
    client.close()

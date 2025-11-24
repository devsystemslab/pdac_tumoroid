import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pandas as pd
import socket
from dask.distributed import Client
from dask_jobqueue import LSFCluster

def setup_dask_client(n_cores:int = 6,
                      n_processes:int = 6,
                      walltime:str = '6:00',
                      memory:str = '16GB',
                      n_jobs:int = 20, gpu:bool = False,
                      task_name:str = 'dask-worker',
                      log_directory:str = None) -> tuple:
    """
    Set up dask client
    :param n_cores:
    :param n_processes:
    :param walltime:
    :param memory:
    :param n_jobs:
    :param gpu:
    :param task_name:
    :param log_directory:
    :return:
    """
    # get hostname
    host_name = socket.gethostname()
    if host_name.__contains__('rkanc'):
        # set up dask cluster on sHPC
        if log_directory is None:
            log_directory = os.getcwd()
        else:
            Path(log_directory).mkdir(parents=True, exist_ok=True)
        # set up extra directives
        extra_directives = [f'-R "rusage[mem={memory}/host]"',
                            f'-o {log_directory}/{task_name}-%J.out',
                            f'-e {log_directory}/{task_name}-%J.err']
        if gpu:
            extra_directives.append('-gpu "num=1:j_exclusive=yes"')
        # set up dask cluster
        cluster = LSFCluster(queue='long',
                             memory=memory,
                             cores=n_cores,
                             processes=n_processes,
                             walltime=walltime,
                             job_extra_directives=extra_directives,
                             job_directives_skip=['-R'])
        cluster.scale(jobs=n_jobs)
        client = Client(cluster)

        print('Dask cluster set up on sHPC and dashboard available at:')
        print(client.dashboard_link)
        return client, cluster

    else:
        client = Client()
        print('Local Dask cluster set up and dashboard available at:')
        print(client.dashboard_link)
        return client, None


def load_plate_information(dir_images: str,
                           dir_processed: str,
                           screen: str,
                           plate: str,
                           dir_raw_imgs: str,
                           **kwargs) -> pd.DataFrame:
    """
    Load plate information
    :param dir_images:
    :param dir_processed:
    :param screen:
    :param plate:
    :param dir_raw_imgs:
    :param kwargs:
    :return:
    """
    # generate data fram columns
    columns = ['screen','plate','plate_id','directory_raw','dir_processed','directory','dir_raw',
               'tile_overlap','cycle','ch01','ch02','ch03','ch04']
    df = pd.DataFrame(columns=columns)
    # add empty rows
    df = df.append(pd.Series(), ignore_index=True)
    if kwargs is not None:
        for key, value in kwargs.items():
            if key in df.columns:
                df[key] = value
    # add plate
    df.loc[:, 'plate'] = plate
    # add dir_raw
    df.loc[:, 'directory'] = dir_raw_imgs
    # add dir_raw
    df.loc[:, 'directory_raw'] = Path(dir_images, dir_raw_imgs)
    if len(os.listdir(Path(dir_images, dir_raw_imgs))) == 1:
        df.loc[:, 'dir_raw'] = Path(df['directory_raw'].values[0], os.listdir(Path(dir_images, dir_raw_imgs))[0])
    else:
        # dir_raw same as directory_raw
        df.loc[:, 'dir_raw'] = Path(dir_images, dir_raw_imgs)
    # add dir_processed
    df.loc[:, 'dir_processed'] = dir_processed
    # add screen to df
    df.loc[:, 'screen'] = screen
    df.loc[:, 'dir_processed'] = df.loc[:, 'plate_id'].apply(lambda x: Path(dir_processed,
                                                                            screen, plate, x))
    return df

def get_metadata(dir_images:str) -> pd.DataFrame:
    """
    Get metadata from image filenames
    :param dir_images:
    :return:
    """
    images = os.listdir(dir_images)
    images = [image for image in images if '.tif' in image]
    regex = r'_(?P<well_id>[A-Z]\d{2})_T(?P<time_point>\d{4})F(?P<field_id>\d{3})L(?P<time_line_id>\d{2,3})A(?P<action_id>\d{2})Z(?P<z_stack_id>\d{2})C(?P<channel_id>\d{2})\.tif$'
    df = pd.DataFrame({'file': images, 'dir_images': str(dir_images)})
    df = df.join(df['file'].str.extractall(regex).groupby(level=0).last())
    # remove rows that have nan in any column
    df = df[~df.isna().any(axis=1)]
    return df

def scale_image(image:np.ndarray, percentile:int = 1, range:tuple = (0,65535)) -> np.ndarray:
    """
    Scale image
    :param image:
    :param percentile:
    :param range:
    :return:
    """
    image = np.interp(image, (np.percentile(image,percentile), np.percentile(image,100 - percentile)), range)
    return image

def show_image(image:np.ndarray, scale:bool = True,
               grayscale:bool = False, adapt_figsize:bool = True,
               factor_adapt:int = 200) -> None:
    """
    Show image
    :param image:
    :param scale:
    :param grayscale:
    :param adapt_figsize:
    :param factor_adapt:
    :return:
    """
    if scale:
        image = scale_image(image)
    if adapt_figsize:
        width = round(image.shape[1]/factor_adapt)
        height = round(image.shape[0]/factor_adapt)
        plt.figure(figsize=(width,height))
    else:
        plt.figure(figsize=(20, 20))
    if grayscale:
        plt.imshow(image, cmap=plt.cm.gray)
    else:
        plt.imshow(image)
    plt.show()


def annotate_img(img: np.ndarray, annotation: str,
                 size: int = 128, x: int = 0, y: int = 0,
                 position:str = 'bottom',color:str = 'white') -> np.ndarray:
    """
    Annotate image
    :param img:
    :param annotation:
    :param size:
    :param x:
    :param y:
    :param position:
    :param color:
    :return:
    """
    fig = plt.figure()
    fig.figimage(
        img,
        resize=True,  # Resize the figure to the image to avoid any interpolation.
    )
    fig.text(x, y, annotation, fontsize=size, color=color, va=position)
    canvas = plt.gca().figure.canvas
    canvas.draw()
    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    annotated_img = data.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return annotated_img

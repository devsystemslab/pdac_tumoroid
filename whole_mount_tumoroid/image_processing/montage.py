from pathlib import Path
from skimage import io
from skimage.util import montage
import numpy as np
from whole_mount_tumoroid.image_processing.utils import get_metadata, scale_image
import pandas as pd

def load_plate_layout(file_plate_layout:str):
    df_layout = pd.read_csv(file_plate_layout)
    # convert plateRow to Letter A-H
    df_layout['plateRow'] = df_layout['plateRow'].apply(lambda x: chr(x + 64))
    # add well_id from plateRow and plateColumn
    df_layout['well_id'] = df_layout['plateRow'] + df_layout['plateColumn'].astype(str).str.zfill(2)
    # arrange by well_id
    df_layout = df_layout.sort_values(by=['well_id'])
    return df_layout


def load_well(well_id:str, df_images:pd.DataFrame, dir_images:str):
    """
    Load well image
    :param well_id:
    :param df_images:
    :param dir_images:
    :return:
    """
    df_images_select = df_images[(df_images['well_id'] == well_id)]
    if len(df_images_select) == 0:
        return None
    elif len(df_images_select) == 1:
        img = io.imread(Path(dir_images, df_images_select['file'].to_list()[0]))
        img = scale_image(img)
        img = img.astype(np.uint16)
    else:
        raise ValueError(f'Well {well_id} has more than one image')
    return img[::8,::8]

def rgb_overlay(channels: list, dir_input:str):
    """
    Create RGB overlay from channels
    :param channels:
    :param dir_input:
    :return:
    """
    files = [Path(dir_input, f'montage_ch{channel}.png') for channel in channels]

    # load images
    images = [io.imread(file) for file in files]

    # scale images
    images = [scale_image(image, range=(0, 1), percentile=0) for image in images]

    # create RGB overlay
    return np.dstack(images)

def generate_montages(dir_plate:str, file_plate_layout:str, input_type:str = 'TIF_MIP_OVR'):
    """
    Generate montages for each channel and overlay
    :param dir_plate:
    :param file_plate_layout:
    :param input_type:
    :return:
    """
    # set up output directory
    dir_output = Path(dir_plate, 'montage', input_type)
    dir_output.mkdir(parents=True, exist_ok=True)
    # set up input directory
    dir_images = Path(dir_plate, input_type)
    df_images = get_metadata(dir_images)
    # get channels and wells
    channels = df_images['channel_id'].unique().tolist()
    df_layout = load_plate_layout(file_plate_layout)
    wells = df_layout['well_id'].to_list()
    # for each well and channel run max projection and create montage
    for channel in channels:
        df_images_select = df_images[(df_images['channel_id'] == channel)]
        df_images_select = df_images_select.sort_values(by=['well_id'])
        imgs = [load_well(well, df_images_select, dir_images) for well in wells]
        # handle missing images
        shapes = tuple(set([img.shape for img in imgs if img is not None]))
        if len(shapes) == 1:
            # replace None with np.zeros(shapes[0])
            imgs = [img if img is not None else np.zeros(shapes[0]) for img in imgs]
        else:
            raise ValueError(f'Images for channel {channel} have different shapes {shapes}')
        file = Path(dir_output,f'montage_ch{channel}.png')
        montage_img = montage(np.asarray(imgs), grid_shape=(16, 24))
        io.imsave(file, scale_image(montage_img, range=(0, 255), percentile=0.1).astype(np.uint8))
    # free memory
    del montage_img
    del imgs
    # generate four stain overlay
    img = rgb_overlay(['02', '03', '04'],  dir_input=dir_output)
    img_2 = rgb_overlay(['01', '01', '01'], dir_input=dir_output)
    img = (img + img_2) / 2
    img = scale_image(img, range=(0, 1), percentile=0)
    # save image
    file = Path(dir_output,f'montage_overlay.png')
    io.imsave(file, img)

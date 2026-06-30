from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm

from analysis.timecourse.plot_tumoroid_examples import add_scalebar
from image_processing.montage import load_well, rgb_overlay
from image_processing.utils import get_metadata, scale_image


def load_plate_layout(file_plate_layout: str):
    df_plate_layout = pd.read_csv(file_plate_layout)
    df_plate_layout = df_plate_layout.melt(
        id_vars=["row"], var_name="col", value_name="staining_set"
    )
    df_plate_layout["col"] = df_plate_layout["col"].str.zfill(2)
    df_plate_layout["row_num"] = df_plate_layout["row"].apply(
        lambda x: ord(x) - ord("A") + 1
    )
    df_plate_layout["well_id"] = df_plate_layout["row"] + df_plate_layout["col"]
    df_plate_layout = df_plate_layout.sort_values(by=["well_id"])

    return df_plate_layout


def generate_stainingset_montage(
    dir_screen: str,
    file_plate_layout: str,
    plates: list,
    staining_set: int,
    input_type: str = "TIF_MIP_OVR_BG",
    scale_bar=(39, 5, 12, 12),
):
    """
    Generate montages for each channel and overlay
    :param dir_plate:
    :param file_plate_layout:
    :param input_type:
    :return:
    """

    plate_montages = []
    dir_output = Path(dir_screen, "montage_stainingsets", str(staining_set))

    for plate in tqdm(
        plates, desc=f"Processing plates for staining set {staining_set}"
    ):
        # set up output directory
        dir_plate = Path(dir_output, plate)
        dir_plate.mkdir(parents=True, exist_ok=True)

        # set up input directory
        dir_images = Path(dir_screen, plate, f"{plate}-01", input_type)
        df_images = get_metadata(dir_images)
        # get channels and wells
        channels = df_images["channel_id"].unique().tolist()
        df_plate_layout = load_plate_layout(file_plate_layout)
        wells = (
            df_plate_layout[df_plate_layout["staining_set"] == staining_set]["well_id"]
            .unique()
            .tolist()
        )

        df_images = df_images.merge(df_plate_layout, how="left", on=["well_id"])
        # for each well and channel run max projection and create montage
        for channel in channels:
            df_images_select = df_images[(df_images["channel_id"] == channel)]
            df_images_select = df_images_select[
                (df_images["staining_set"] == staining_set)
            ]
            df_images_select = df_images_select.sort_values(by=["well_id"])
            imgs = [load_well(well, df_images_select, dir_images) for well in wells]
            # handle missing images
            shapes = tuple(set([img.shape for img in imgs if img is not None]))
            if len(shapes) == 1:
                # replace None with np.zeros(shapes[0])
                imgs = [img if img is not None else np.zeros(shapes[0]) for img in imgs]
            else:
                raise ValueError(
                    f"Images for channel {channel} have different shapes {shapes}"
                )
            file = Path(dir_plate, f"montage_ch{channel}.png")
            montage_img = np.vstack(imgs)
            io.imsave(
                file,
                scale_image(montage_img, range=(0, 255), percentile=0.1).astype(
                    np.uint8
                ),
            )
        # free memory
        del montage_img
        del imgs
        # generate four stain overlay

        img = rgb_overlay(["02", "03", "04"], dir_input=dir_plate)
        img_2 = rgb_overlay(["01", "01", "01"], dir_input=dir_plate)
        img = (img + img_2) / 2
        img = scale_image(img, range=(0, 255), percentile=0).astype(np.uint8)

        plate_montages.append(img)
        # save image
        file = Path(dir_plate, f"montage_overlay.png")
        io.imsave(file, img)

    staining_set_timecourse_montage = np.hstack(plate_montages)
    file = Path(dir_output, f"stainingset_{staining_set}_montage.png")

    width, height, offset_x, offset_y = scale_bar

    staining_set_timecourse_montage = add_scalebar(
        staining_set_timecourse_montage,
        width=width,
        height=height,
        x=staining_set_timecourse_montage.shape[1] - offset_x - width,
        y=staining_set_timecourse_montage.shape[0] - offset_y - height,
    )

    io.imsave(file, staining_set_timecourse_montage)


if __name__ == "__main__":
    dir_screen = "data/timecourse"
    file_plate_layout = "metafiles/timecourse_layout.csv"

    plates = ["001", "002", "003", "004", "005"]
    staining_sets = list(range(1, 13))

    for stain_set in staining_sets:
        generate_stainingset_montage(dir_screen, file_plate_layout, plates, stain_set)

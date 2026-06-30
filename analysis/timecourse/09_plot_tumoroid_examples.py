from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from skimage.exposure import rescale_intensity
from skimage.util import dtype_limits

from phenocoder.plot import RGBRotate
from phenocoder.utils import get_metadata


def scale_image(
    image: np.ndarray,
    percentiles: tuple = (1, 99),
    range: tuple = (0, 65535),
) -> np.ndarray:
    """
    Scale image
    :param image:
    :param percentiles:
    :param range:
    :return:
    """
    image = np.interp(
        image,
        (np.percentile(image, percentiles[0]), np.percentile(image, percentiles[1])),
        range,
    )
    return image


def plot_channels(imgs: np.ndarray, df_images: pd.DataFrame) -> None:
    """
    Plot channels
    :param imgs:
    :param df_images:
    :return:
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, ax in enumerate(axs.flat):
        ax.imshow(imgs[i])
        ax.set_title(df_images["channel_id"].values[i])
        ax.axis("off")
    plt.show()


def add_scalebar(
    img: np.ndarray, width: int, height: int, x: int, y: int
) -> np.ndarray:
    """
    Add scale bar to image
    :param img:
    :param width:
    :param height:
    :param x:
    :param y:
    :return:
    """
    img = img.copy()
    # get dtype max
    max_val = dtype_limits(img)[1]
    img[y : y + height, x : x + width, :] = max_val
    return img


def overlay_colors(imgs):
    """
    Overlay colors
    :param imgs:
    :return:
    """
    imgs = np.moveaxis(imgs, 0, -1)
    img = imgs[..., 1:]
    img_2 = imgs[..., 0]
    # duplicate img2 3 times in axis
    img_2 = np.repeat(img_2[:, :, np.newaxis], 3, axis=2)
    img = (img + img_2) / 2
    img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
    # hue rotation
    rotator = RGBRotate()
    rotator.set_hue_rotation(49)
    img = rotator.apply_to_image(img)
    img = scale_image(img, range=(0, 255)).astype(np.uint8)
    return img


# construct dataframe from wells and timepoints
def plot_organoid(
    well,
    plate,
    luts,
    save,
    plot,
    timepoints,
    img_type,
    dir_output=None,
    channel_order=None,
    scale_bar=(310, 40, 100, 100),
):
    """
    Plot organoid
    :param well:
    :param plate:
    :param luts:
    :param save:
    :param plot:
    :param dir_output:
    :param scale_bar:
    :return:
    """
    df_images = get_metadata(Path(dir_screen, plate, f"{plate}-01", img_type))
    df_images = df_images[df_images["well_id"] == well]
    if df_images.shape[0] == 4:
        if channel_order is None:
            channel_order = ["01", "02", "03", "04"]
        # sort by channel_order
        df_images["channel_id"] = pd.Categorical(
            df_images["channel_id"], categories=channel_order, ordered=True
        )
        df_images = df_images.sort_values(by="channel_id")
        imgs = np.asarray(
            [
                io.imread(Path(dir_screen, plate, f"{plate}-01", img_type, file))
                for file in df_images["file"].values
            ]
        )
        imgs = np.asarray(
            [
                scale_image(imgs[i], percentiles=luts[i], range=(0, 1))
                for i in range(imgs.shape[0])
            ]
        )
        imgs_overlay = overlay_colors(imgs)
        if scale_bar is not None:
            width, height, offset_x, offset_y = scale_bar
            imgs = np.moveaxis(imgs, 0, -1)
            imgs = add_scalebar(
                imgs,
                width=width,
                height=height,
                x=imgs.shape[1] - offset_x - width,
                y=imgs.shape[0] - offset_y - height,
            )
            imgs = np.moveaxis(imgs, -1, 0)
            imgs_overlay = add_scalebar(
                imgs_overlay,
                width=width,
                height=height,
                x=imgs.shape[1] - offset_x - width,
                y=imgs.shape[0] - offset_y - height,
            )
        if plot:
            plot_channels(imgs, df_images)
            fig = plt.figure(figsize=(10, 10))
            io.imshow(imgs_overlay)
            plt.show()
            plt.close("all")
        if save:
            imgs_overlay = imgs_overlay[::2, ::2, :]
            imgs = imgs[:, ::2, ::2]
            if dir_output is None:
                raise ValueError("dir_output is None")
            else:
                io.imsave(
                    Path(dir_output, f"{well}_{timepoints[plate]}.png"), imgs_overlay
                )
                for i, channel in enumerate(channel_order):
                    io.imsave(
                        Path(
                            dir_output, f"{well}_{timepoints[plate]}_ch_{channel}.png"
                        ),
                        rescale_intensity(imgs[i], out_range=(0, 255)).astype(np.uint8),
                    )
    else:
        print(f"Well {well} timepoints {timepoints[plate]} is incomplete.")


if __name__ == "__main__":
    dir_screen = "data/timecourse"
    df_stains = pd.read_csv(
        "metafiles/timecourse_stainings_metadata.csv"
    )
    df_plate_layout = pd.read_csv(
        "metafiles/timecourse_layout.csv"
    )
    df_plate_layout = df_plate_layout.melt(
        id_vars=["row"], var_name="col", value_name="staining_set"
    )
    df_plate_layout["col"] = df_plate_layout["col"].str.zfill(2)
    df_plate_layout["well"] = df_plate_layout["row"] + df_plate_layout["col"]
    df_plate_layout = df_plate_layout.merge(df_stains, on=["staining_set"])
    df_plate_layout["channel"] = df_plate_layout["channel"].astype(str).str.zfill(2)

    def generate_images_for_staining_set(
        wells: dict,
        timepoints: dict,
        luts: dict,
        dir_output: str,
        img_type="TIF_MIP_OVR_BG",
        channel_order=None,
        save=False,
        plot=True,
    ):
        """
        Generate images for staining set
        :param wells:
        :param timepoints:
        :param luts:
        :param dir_output:
        :param img_type:
        :param channel_order:
        :param save:
        :param plot:
        :return:
        """
        dir_output = Path(dir_output)
        dir_output.mkdir(parents=True, exist_ok=True)
        df_wells = pd.DataFrame(
            {"plate": list(wells.keys()), "well": list(wells.values())}
        )
        df_timepoints = pd.DataFrame(
            {"plate": list(timepoints.keys()), "timepoint": list(timepoints.values())}
        )
        df = df_wells.merge(df_timepoints, on="plate", how="left")
        for i, row in df.iterrows():
            plot_organoid(
                well=row["well"],
                plate=row["plate"],
                luts=luts[row["plate"]],
                timepoints=timepoints,
                plot=plot,
                img_type=img_type,
                channel_order=channel_order,
                save=save,
                dir_output=dir_output,
            )

    # assign plates to timepoints
    timepoints = {
        "001": "day3",
        "002": "day5",
        "003": "day7",
        "004": "day10",
        "005": "day14",
    }
    luts = {
        "001": [(1, 99), (40, 95), (1, 99), (0.25, 99.75)],
        "002": [(1, 99), (20, 90), (1, 99), (0.25, 99.75)],
        "003": [(0, 100), (20, 95), (0, 100), (1, 99)],
        "004": [(1, 99), (1, 99), (1, 99), (1, 99)],
        "005": [(1, 99), (1, 99), (1, 99), (1, 99)],
    }
    dir_output = "data/timecourse/example_overlays_SDC1_YAP_Phalloidin"
    # generate overlays for DAPI, SDC1, Phalloidin and YAP
    wells = {"001": "H05", "002": "H05", "003": "H05", "004": "H05", "005": "D05"}
    generate_images_for_staining_set(
        wells, timepoints, luts, dir_output, save=False, plot=True
    )

    dir_output = "data/timecourse/example_overlays_SDC1_ITGA2_LAMC2"
    # generate overlays for DAPI, SDC1, Phalloidin and YAP
    well_positions = ["A01", "B01", "C01", "D01", "E01", "F01", "G01"]

    for well in well_positions:
        wells = {"001": well, "002": well, "003": well, "004": well, "005": well}
        luts = {
            "001": [(1, 99), (1, 99), (1, 99), (1, 99)],
            "002": [(1, 99), (1, 99), (1, 99), (1, 99)],
            "003": [(1, 99), (1, 99), (1, 99), (1, 99)],
            "004": [(1, 99), (1, 99), (1, 99), (1, 99)],
            "005": [(1, 99), (1, 99), (1, 99), (1, 99)],
        }
        generate_images_for_staining_set(
            wells, timepoints, luts, dir_output, save=True, plot=False
        )

    dir_output = "data/timecourse/example_overlays_COL1_KI67_CK19"
    well_positions = ["I01", "J01", "K01", "L01", "M01", "N01", "O01"]

    for well in well_positions:
        wells = {"001": well, "002": well, "003": well, "004": well, "005": well}
        luts = {
            "001": [(1, 99), (5, 99), (90, 99), (1, 99)],
            "002": [(1, 99), (5, 99), (90, 99), (1, 99)],
            "003": [(1, 99), (5, 99), (90, 99), (1, 99)],
            "004": [(1, 99), (5, 99), (90, 99), (1, 99)],
            "005": [(1, 99), (5, 99), (90, 99), (1, 99)],
        }
        generate_images_for_staining_set(
            wells, timepoints, luts, dir_output, save=True, plot=False
        )

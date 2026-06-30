from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io
from skimage.exposure import adjust_gamma, rescale_intensity

from analysis.utils import add_scalebar
from phenocoder.plot import RGBRotate
from phenocoder.utils import get_metadata

if __name__ == "__main__":
    # C11 - staining test- DAPI, SDC1, ITGA2, LAMC2
    dir_images = "data/staining_test/001/001-01"
    df_images = get_metadata(Path(dir_images, "TIF_OVR"))
    # filter for well C11
    df_images = df_images[df_images["well_id"] == "C11"]
    # filter for channel_id in '01' -'04'
    df_images = df_images[df_images["channel_id"].isin(["01", "02", "03", "04"])]
    # sort by channel_id and z_stack_id
    df_images = df_images.sort_values(["channel_id", "z_stack_id"])
    # for each channel read in z_stack and concatenate channels on first axis, z-axis on second
    imgs = []
    for channel in ["01", "02", "03", "04"]:
        imgs.append(
            np.asarray(
                [
                    io.imread(Path(dir_images, "TIF_OVR", file))
                    for file in df_images[df_images["channel_id"] == channel]["file"].values
                ]
            )
        )
    imgs = np.asarray(imgs)

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
            overlap_img = (img_ref[-overlap:, :] * blend) + (img_add[:overlap, :] * (1 - blend))
            img = np.concatenate([img_ref[:-overlap, :], overlap_img, img_add[overlap:, :]], axis=axis)
        elif axis == 1:
            # make 2D array from blend
            blend = np.tile(blend, (shape[1 - axis], 1))
            overlap_img = (img_ref[:, -overlap:] * blend) + (img_add[:, :overlap] * (1 - blend))
            img = np.concatenate([img_ref[:, :-overlap], overlap_img, img_add[:, overlap:]], axis=axis)
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
        else:
            raise ValueError("Number of images not 4.")
        return img

    def split_img(img):
        """
        Split image into 4 quadrants
        :param img: image to split
        :return: list of 4 images
        """
        h, w = img.shape[:2]
        return np.asarray(
            [
                img[: h // 2, : w // 2],
                img[: h // 2, w // 2 :],
                img[h // 2 :, : w // 2],
                img[h // 2 :, w // 2 :],
            ]
        )

    img_split = split_img(imgs[0, 10, ...])
    img_corr = stitch_montage(img_split, overlap=0.01)
    plt.figure(figsize=(20, 20))
    io.imshow(img_corr)
    plt.show()

    imgs_all = []
    for i in range(imgs.shape[0]):
        imgs_corr = []
        for j in range(imgs.shape[1]):
            img_split = split_img(imgs[i, j, ...])
            img_corr = stitch_montage(img_split, overlap=0.01)
            imgs_corr.append(img_corr)
        imgs_corr = np.asarray(imgs_corr)
        imgs_all.append(imgs_corr)
    imgs_all = np.asarray(imgs_all)
    imgs = imgs_all.copy()

    def scale_image(
        image: np.ndarray,
        percentiles: tuple[int, int] = (1, 99),
        range: tuple[int, int] = (0, 65535),
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
            (
                np.percentile(image, percentiles[0]),
                np.percentile(image, percentiles[1]),
            ),
            range,
        )
        return image

    def overlay_colors(imgs, four_channel=False, rotate=False):
        """
        Overlay colors
        :param imgs:
        :return:
        """
        imgs = np.moveaxis(imgs, 0, -1)
        if four_channel:
            img = imgs[..., 1:]
            img_2 = imgs[..., 0]
            # duplicate img2 3 times in axis
            img_2 = np.repeat(img_2[:, :, np.newaxis], 3, axis=2)
            img = (img + img_2) / 2
        else:
            img = imgs.copy()
        img = rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        if rotate:  # hue rotation
            rotator = RGBRotate()
            rotator.set_hue_rotation(49)
            img = rotator.apply_to_image(img)
            # img = scale_image(img, range=(0,255)).astype(np.uint8)
            img = img.astype(np.uint8)
        return img

    def get_overlay(imgs, luts, gammas, rotate=True, four_channel=False):
        imgs = np.asarray([adjust_gamma(imgs[i], gammas[i]) for i in range(imgs.shape[0])])
        imgs = np.asarray([scale_image(imgs[i], percentiles=luts[i], range=(0, 1)) for i in range(imgs.shape[0])])
        imgs_overlay = overlay_colors(imgs, rotate=rotate, four_channel=four_channel)

        return imgs_overlay

    luts = [(1, 99.5), (1, 99.5), (0.5, 99.75)]
    gammas = [1, 1, 1.5]
    img_overlay = get_overlay(imgs[1:].max(axis=1)[:, 750:, 800:], luts, gammas)
    width, height, offset_x, offset_y = (310, 40, 100, 100)
    img_overlay = add_scalebar(
        img_overlay,
        width=width,
        height=height,
        x=img_overlay.shape[1] - offset_x - width,
        y=img_overlay.shape[0] - offset_y - height,
    )
    fig = plt.figure(figsize=(20, 20))
    io.imshow(img_overlay)
    plt.show()
    # save image:
    io.imsave(
        "data/staining_test/001/tumoroid_viewer/screenshots/C11_overlay_python.png",
        img_overlay,
    )

    luts = [(1, 99.9), (1, 99.9), (0.5, 99.975)]
    gammas = [1.2, 1.2, 1.5]
    img_overlay = get_overlay(imgs[1:, 14, 750:, 800:], luts, gammas)
    width, height, offset_x, offset_y = (310, 40, 100, 100)
    img_overlay = add_scalebar(
        img_overlay,
        width=width,
        height=height,
        x=img_overlay.shape[1] - offset_x - width,
        y=img_overlay.shape[0] - offset_y - height,
    )
    fig = plt.figure(figsize=(20, 20))
    io.imshow(img_overlay)
    plt.show()
    # save image:
    io.imsave(
        "data/staining_test/001/tumoroid_viewer/screenshots/C11_overlay_python_z14.png",
        img_overlay,
    )

    # H08 - pilotscreen dataset - DAPI, SDC1, ITGA2, LAMC2
    dir_images = "data/pilotscreen/004/004-03"
    df_images = get_metadata(Path(dir_images, "TIF_OVR"))
    # filter for well C11
    df_images = df_images[df_images["well_id"] == "H08"]
    # filter for channel_id in '01' -'04'
    df_images = df_images[df_images["channel_id"].isin(["01", "02", "03", "04"])]
    # sort by channel_id and z_stack_id
    df_images = df_images.sort_values(["channel_id", "z_stack_id"])
    # for each channel read in z_stack and concatenate channels on first axis, z-axis on second
    imgs = []
    for channel in ["01", "02", "03", "04"]:
        imgs.append(
            np.asarray(
                [
                    io.imread(Path(dir_images, "TIF_OVR", file))
                    for file in df_images[df_images["channel_id"] == channel]["file"].values
                ]
            )
        )
    imgs = np.asarray(imgs)

    luts = [(1, 99), (30, 95), (92, 99.9), (5, 95)]
    gammas = [1, 0.25, 1, 0.75]
    img_overlay = get_overlay(imgs[:, 2:, 500:3400, 500:3400].max(axis=1), luts, gammas, four_channel=True)
    width, height, offset_x, offset_y = (310, 40, 100, 100)
    img_overlay = add_scalebar(
        img_overlay,
        width=width,
        height=height,
        x=img_overlay.shape[1] - offset_x - width,
        y=img_overlay.shape[0] - offset_y - height,
    )
    fig = plt.figure(figsize=(20, 20))
    io.imshow(img_overlay)
    plt.show()

    io.imsave(
        "data/pilotscreen/example_overlays/004-03/H08/overlay_figure_2.png",
        img_overlay,
    )
    for i in range(30):
        luts = [(0, 99.9), (10, 99), (40, 99.9), (0.1, 99.9)]
        gammas = [1.5, 0.75, 1.5, 1]
        img_overlay = get_overlay(imgs[:, i, 500:3400, 500:3400], luts, gammas, four_channel=True)
        width, height, offset_x, offset_y = (310, 40, 100, 100)
        img_overlay = add_scalebar(
            img_overlay,
            width=width,
            height=height,
            x=img_overlay.shape[1] - offset_x - width,
            y=img_overlay.shape[0] - offset_y - height,
        )
        # fig = plt.figure(figsize=(20, 20))
        # io.imshow(img_overlay)
        # plt.show()
        io.imsave(
            f"data/pilotscreen/example_overlays/004-03/H08/overlay_figure_2_zslice_{i}.png",
            img_overlay,
        )
    img_collagen = scale_image(
        imgs[:, 2:, 500:3400, 500:3400].max(axis=1)[1],
        range=(0, 255),
        percentiles=(20, 99),
    ).astype("uint8")
    img_collagen = adjust_gamma(img_collagen, 1.2)
    img_collagen = add_scalebar(
        np.repeat(img_collagen[:, :, np.newaxis], 3, axis=2),
        width=width,
        height=height,
        x=img_overlay.shape[1] - offset_x - width,
        y=img_overlay.shape[0] - offset_y - height,
    )
    io.imsave(
        "data/pilotscreen/example_overlays/004-03/H08/overlay_figure_2_collagen.png",
        img_collagen,
    )

    def plot_timecourse_example(
        well,
        plate,
        plot=True,
        save=False,
        luts=[(1, 99), (1, 99), (1, 99), (1, 99)],
        gammas=[1, 1, 1, 1],
        prefix=None,
    ):
        # H05 - timecourse
        dir_images = f"data/timecourse/{plate}/{plate}-01"
        df_images = get_metadata(Path(dir_images, "TIF_OVR_BG"))
        # filter for well C11
        df_images = df_images[df_images["well_id"] == well]
        # filter for channel_id in '01' -'04'
        df_images = df_images[df_images["channel_id"].isin(["01", "02", "03", "04"])]
        # sort by channel_id and z_stack_id
        df_images = df_images.sort_values(["channel_id", "z_stack_id"])
        # for each channel read in z_stack and concatenate channels on first axis, z-axis on second
        imgs = []
        for channel in ["01", "02", "03", "04"]:
            imgs.append(
                np.asarray(
                    [
                        io.imread(Path(dir_images, "TIF_OVR", file))
                        for file in df_images[df_images["channel_id"] == channel]["file"].values
                    ]
                )
            )
        imgs = np.asarray(imgs)

        img_overlay = get_overlay(imgs[1:, ...].max(axis=1), luts, gammas, four_channel=False)
        width, height, offset_x, offset_y = (310, 40, 100, 100)
        img_overlay = add_scalebar(
            img_overlay,
            width=width,
            height=height,
            x=img_overlay.shape[1] - offset_x - width,
            y=img_overlay.shape[0] - offset_y - height,
        )
        if plot:
            fig = plt.figure(figsize=(20, 20))
            io.imshow(img_overlay)
            plt.show()
        if save:
            io.imsave(
                f"data/timecourse/example_overlays_figure/{prefix}-{plate}-{well}.png",
                img_overlay,
            )

    dir_screen = "data/timecourse"
    df_stains = pd.read_csv("metafiles/timecourse_stainings_metadata.csv")
    df_plate_layout = pd.read_csv("metafiles/timecourse_layout.csv")
    df_plate_layout = df_plate_layout.melt(id_vars=["row"], var_name="col", value_name="staining_set")
    df_plate_layout["col"] = df_plate_layout["col"].str.zfill(2)
    df_plate_layout["well"] = df_plate_layout["row"] + df_plate_layout["col"]
    df_plate_layout = df_plate_layout.merge(df_stains, on=["staining_set"])
    df_plate_layout["channel"] = df_plate_layout["channel"].astype(str).str.zfill(2)

    # staining set 1
    wells = {"001": "H01", "002": "F01", "003": "C01", "004": "A01", "005": "A0"}
    for plate, well in wells.items():
        plot_timecourse_example(well, plate, False, True, prefix="set_1")
        print(f"{plate}-{well} processed.")

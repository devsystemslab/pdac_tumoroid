import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# load plate
def generate_plate_montage(
    plate,
    dir_images,
    dir_output,
    annotate=False,
    down_sample_factor=None,
    shape=(16, 24),
):
    files = sorted(os.listdir(os.path.join(dir_images, plate)))
    # select tif files
    files_wells = [x for x in files if ".tif" in x]
    wells = [file.split("_")[1] for file in files_wells]
    # select first three characters
    wells = [well[:3] for well in wells]
    # sort files wells according to wells
    files_wells = [file for _, file in sorted(zip(wells, files_wells))]
    # sort wells
    wells = sorted(wells)
    # read in images
    imgs = [
        io.imread(os.path.join(dir_images, plate, file))
        for file in tqdm(files_wells, desc="Loading images", total=len(files_wells))
    ]
    imgs = [
        scale_image(img) for img in tqdm(imgs, desc=f"Scaling images", total=len(imgs))
    ]
    # annotate each image
    if annotate:
        imgs = [
            annotate_img(img, annotation=well, size=75)
            for img, well in tqdm(
                zip(imgs, wells), desc="Annotating images", total=len(imgs)
            )
        ]
        # convert to grayscale
        imgs = [
            rgb2gray(img)
            for img in tqdm(imgs, desc="Converting to grayscale", total=len(imgs))
        ]
    # downsample
    if down_sample_factor is not None:
        imgs = [
            img[::down_sample_factor, ::down_sample_factor]
            for img in tqdm(imgs, desc=f"Down sampling images", total=len(imgs))
        ]
    print(f"Generating montage...")
    imgs = np.asarray(imgs)
    img = montage(imgs, grid_shape=shape)
    # convert to uint8
    img = (img / img.max() * 255).astype(np.uint8)
    print(f"Saving image...")
    io.imsave(f"{dir_output}/{plate}.png", img)


def generate_timecourse_montage(
    well, dir_images, dir_output, down_sample_factor=None, shape=(2, 6)
):
    timepoints = sorted(os.listdir(dir_images))
    imgs = []
    for timepoint in tqdm(timepoints, desc="Processing images", total=len(timepoints)):
        files = sorted(os.listdir(os.path.join(dir_images, timepoint)))
        wells = [well.split("_")[1] for well in files]
        wells = [well[:3] for well in wells]
        # get index of matching well
        file = files[wells.index(well)]
        img = io.imread(os.path.join(dir_images, timepoint, file))
        img = scale_image(img)
        if down_sample_factor is not None:
            img = img[::down_sample_factor, ::down_sample_factor]
        imgs.append(img)
        # save img to output
        io.imsave(f"{dir_output}/{well}_{timepoint}.png", img)
    imgs = np.asarray(imgs)
    img = montage(imgs, grid_shape=shape, fill=imgs.max())
    # convert to uint8
    img = (img / img.max() * 255).astype(np.uint8)
    print(f"Saving image...")
    io.imsave(f"{dir_output}/{well}_timecourse.png", img)


if __name__ == "__main__":
    # 4 plates - day 3
    dir_images = "MD_ImageXpress/2250729IHBOD081RO001"
    # set the directory
    plates = sorted(os.listdir(dir_images))
    plates = [file for file in plates if not file.startswith(".")]
    dir_output = "data/bf_imaging/montages"
    for plate in plates:
        generate_plate_montage(
            plate, dir_images, dir_output, annotate=False, down_sample_factor=4
        )

    # timecourse starting from day 3 - 14
    dir_images = "MD_ImageXpress/2250721IHBOD081RO001"
    # set the directory
    plates = sorted(os.listdir(dir_images))
    plates = [file for file in plates if not file.startswith(".")]
    dir_output = "data/bf_imaging/montages"
    for plate in plates:
        generate_plate_montage(
            plate,
            dir_images,
            dir_output,
            annotate=False,
            down_sample_factor=4,
            shape=(16, 6),
        )

    generate_timecourse_montage("E05", dir_images, dir_output, down_sample_factor=None)

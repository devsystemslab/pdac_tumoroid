from math import cos, radians, sin
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from skimage import measure
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian, threshold_otsu
from skimage.transform import rotate
from tqdm import tqdm

from laminator import Laminator


class LaminarWindows(Laminator):
    """Class to extract oriented windows froom images"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.window_width = kwargs.get("window_width", None)
        self.window_height = kwargs.get("window_height", None)
        self.windows = None
        self.angles = None
        self.sample_rate = None
        self.distance_img = None
        self.contours = None
        self._mode = "laminate"

    def set_window_size(self, width: int, height: int) -> Laminator:
        """
        Set window size

        :param width:
        :param height:
        :return:
        """
        self.window_width = width
        self.window_height = height

    def get_contour(self, smoothing: bool = True, sigma: int = 50) -> Laminator:
        """
        Calculate contours from mask

        :param smoothing:
        :param sigma:
        :return:

        """
        # Smooth mask outline
        if smoothing:
            self.mask = gaussian(self.mask, sigma=sigma)
            thr = threshold_otsu(self.mask)
            self.mask = self.mask > thr

        # Find contours
        contours = measure.find_contours(self.mask, 0.5)
        results = []
        for i, contour in enumerate(contours):
            results.append(contour.round().astype(int))

        df = []
        for i in range(len(results)):
            df_tmp = pd.DataFrame(results[i]).assign(contour=i).drop_duplicates()
            df_tmp["position"] = df_tmp.index
            df.append(df_tmp)

        self.contours = pd.concat(df).rename(
            columns={0: "y", 1: "x"}
        )  # unify contour and positions -> mainly the same!

    def distance_transform(self, smoothing: bool = True, sigma: int = 25) -> Laminator:
        """
        Calculate distance transform of mask

        :param self:
        :param smoothing:
        :param sigma:
        :return:
        """
        self.distance_img = distance_transform_edt(self.mask)
        if smoothing:
            self.distance_img = gaussian(self.distance_img, sigma=sigma)

    # LAMINATE

    def assess_rotation(self, angle: int, img: np.ndarray, scale: tuple = (10, 2)) -> float:
        """
        Assess rotation of image

        :param self:
        :param angle:
        :param img:
        :param scale:
        :return:
        """
        range_y = round(self.window_height / scale[0]) + self.window_height
        range_x = round(self.window_width / scale[1])
        rotated_img = rotate(img, float(angle))
        eval_rotation = rotated_img[
            self.window_height : range_y, self.window_height - range_x : self.window_height + range_x
        ]
        eval_rotation = eval_rotation.sum() * (-1)
        return eval_rotation

    def get_angle(self, x: int, y: int, position: int, contour: int) -> pd.DataFrame:
        """
        Get angle of wedge

        :param self:
        :param x:
        :param y:
        :param position:
        :param contour:
        :return:
        """
        # Get boundaries of wedge_area on original image
        x_offset = x - self.window_height
        y_offset = y - self.window_height

        x_offset = np.where(x_offset < 0, np.absolute(x_offset), 0)
        y_offset = np.where(y_offset < 0, np.absolute(y_offset), 0)

        # Crop image centered to point with size that keeps circle of window
        cropped = self.distance_img[
            y - self.window_height + y_offset : y + self.window_height,
            x - self.window_height + x_offset : x + self.window_height,
        ]
        padded = np.zeros([2 * self.window_height, 2 * self.window_height])
        padded[y_offset : cropped.shape[0] + y_offset, x_offset : cropped.shape[1] + x_offset] = cropped

        # Optimize rotation of cropped image for half of the window by summing the corresponding area of distance image
        res = minimize(self.assess_rotation, x0=0, bounds=[(-180, 180)], args=padded)
        angle = res["x"]
        df_angle = pd.DataFrame().assign(
            angle=angle,
            x=x,
            y=y,
            window_height=self.window_height,
            window_width=self.window_width,
            x_offset=x_offset,
            y_offset=y_offset,
            position=position,
            contour=contour,
        )
        return df_angle

    def calculate_angles(self, sample_rate: int = None) -> Laminator:
        """
        Calculate angles of wedges

        :param self:
        :param sample_rate:
        :return:
        """
        if {self.window_height, self.window_width} == {None}:
            raise ValueError("self.window_height and self.window_width must be specified")

        if sample_rate is None:
            self.sample_rate = self.window_width
        else:
            self.sample_rate = sample_rate

        contours = self.contours.iloc[:: self.sample_rate, :]
        results = []
        for x, y, position, contour in tqdm(
            zip(contours["x"].values, contours["y"].values, contours["position"].values, contours["contour"].values),
            total=len(contours),
            desc="Calculating angles",
        ):
            results.append(self.get_angle(x, y, position, contour))

        results = pd.concat(results)
        self.angles = results.assign(window=range(0, len(results)))

    def assess_oriented_windows(
        self,
        img: np.ndarray,
        show_angles: bool = True,
        show_plot: bool = True,
        save_plot: bool = False,
        file: str = None,
        arrow_radius: int = 100,
    ):
        """
        Plot the angles on the original image

        :param img:
        :param show_angles:
        :param show_plot:
        :param save_plot:
        :param file:
        :param arrow_radius:
        :return:
        """
        # Plot contour on original image
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(rescale_intensity(img), cmap=plt.cm.gray)

        if show_angles:
            r = arrow_radius
            for index, row in self.angles.iterrows():
                ax.arrow(
                    row["x"],
                    row["y"],
                    r * cos(radians(row["angle"] + 90)),
                    r * sin(radians(row["angle"] + 90)),
                    color="blue",
                )

        ax.scatter(self.angles["x"].values, self.angles["y"].values, c="red")
        ax.axis("image")
        ax.set_xticks([])
        ax.set_yticks([])
        if save_plot:
            plt.savefig(str(file + ".png"))
        if show_plot:
            plt.show()
        plt.close("all")

    def generate_oriented_window(self, angle: int, x: int, y: int, img: np.ndarray) -> np.ndarray:
        """
        Retrieve oriented window

        :param self:
        :param angle:
        :param x:
        :param y:
        :param img:
        :return:
        """
        window = np.zeros([2 * self.window_height, 2 * self.window_height])
        window[
            self.window_height :, self.window_height - self.window_width : self.window_height + self.window_width
        ] = 1
        window = rotate(window, -angle)
        window_positioned = np.zeros(img.shape)
        x_min = int(np.array([(x - self.window_height) * -1 if (x - self.window_height) < 0 else 0]).astype(int))
        x_max = int(
            np.array(
                [
                    window.shape[1] - (x + self.window_height - img.shape[1])
                    if (x + self.window_height) > img.shape[1]
                    else window.shape[1]
                ]
            ).astype(int)
        )
        y_min = int(np.array([(y - self.window_height) * -1 if (y - self.window_height) < 0 else 0]).astype(int))
        y_max = int(
            np.array(
                [
                    window.shape[0] - (y + self.window_height - img.shape[0])
                    if (y + self.window_height) > img.shape[0]
                    else window.shape[0]
                ]
            ).astype(int)
        )
        window = window[y_min:y_max, x_min:x_max]
        window_positioned[
            y - self.window_height + y_min : y + self.window_height,
            x - self.window_height + x_min : x + self.window_height,
        ] = window
        window_positioned = np.where(window_positioned > 0, 1, 0)
        return window_positioned

    def create_window_stack(self, img: np.ndarray) -> Laminator:
        """
        Create stack of windows for given image

        :param img:
        :return:
        """
        self.windows = []
        for index, row in self.angles.iterrows():
            window_positioned = self.generate_oriented_window(row["angle"], row["x"], row["y"], img)
            self.windows.append(window_positioned)
        self.windows = np.dstack(self.windows)

    def plot_window_stack_collage(
        self, group_cluster: bool = False, subset: list = None, save_plot: bool = False, file: str = None
    ) -> None:
        """
        Plot collage of windows

        :param group_cluster:
        :param subset:
        :param save_plot:
        :param file:
        :return:
        """
        img = np.hstack(self.windows)
        plt.imshow(img, cmap=plt.cm.gray)
        plt.show()

    # TODO: add implementation to generate windows for all images not just for one img
    def get_oriented_intensity_profile(self, angle: int, x: int, y: int, x_offset: int, y_offset: int):
        """
        Get intensity profile of oriented window

        :param angle:
        :param x:
        :param y:
        :param x_offset:
        :param y_offset:
        :return:
        """
        # Crop image centered to point with size that keeps circle of window
        imgs_cropped = self.images[
            y - self.window_height + y_offset : y + self.window_height,
            x - self.window_height + x_offset : x + self.window_height,
            ...,
        ]
        imgs_padded = np.zeros([2 * self.window_height, 2 * self.window_height, self.images.shape[2]])
        imgs_padded[y_offset : imgs_cropped.shape[0] + y_offset, x_offset : imgs_cropped.shape[1] + x_offset, ...] = (
            imgs_cropped
        )

        # Apply rotation to original cropped image
        imgs_rotated = rotate(imgs_padded, angle)[
            self.window_height :, self.window_height - self.window_width : self.window_height + self.window_width, ...
        ]
        imgs_rotated = np.average(imgs_rotated, axis=1)

        return imgs_rotated

    def rotate_windows(self) -> Laminator:
        """
        Rotate windows

        :return:
        """
        if self.angles is None:
            raise ValueError("Angles have not been calculated yet. Run Laminator.calculate_angles() first.")

        result = []
        for angle, x, y, x_offset, y_offset in tqdm(
            zip(
                self.angles["angle"].values,
                self.angles["x"].values,
                self.angles["y"].values,
                self.angles["x_offset"].values,
                self.angles["y_offset"].values,
            ),
            total=self.angles.shape[0],
            desc="Extracting oriented intensity profiles",
        ):
            tmp = self.get_oriented_intensity_profile(angle, x, y, x_offset, y_offset)
            result.append(tmp)

        # Assemble results into numpy stack
        result = np.dstack(result)

        # Convert to pandas dataframe
        names = ["radial_position", "image", "window"]
        index = pd.MultiIndex.from_product([range(s) for s in result.shape], names=names)
        self.intensities = pd.DataFrame({"intensity": result.flatten()}, index=index)["intensity"].reset_index()

        # rename image with self.image_names dict
        if self.image_names is not None:
            self.intensities["image"] = self.intensities["image"].map(self.image_names)

    # WRAPPERS
    def extract_intensities(self) -> Laminator:  # write wrapper for each subclass
        """
        Extract intensities from images

        :return:
        """
        if self.images is None:
            raise ValueError("No images loaded. Run Laminator.load_images() first.")
        if self.mask is None:
            raise ValueError("No mask loaded. Run Laminator.load_mask() first.")
        self.get_contour()
        self.distance_transform(smoothing=True, sigma=10)
        self.calculate_angles()
        self.rotate_windows()
        self.average_intensities()

    def save_results(self, output_path=None) -> None:  # write for each subclass
        """
        Save results to output_path

        :return:
        """
        if self.output_path is None:
            if output_path is None:
                raise ValueError(
                    "No output path specified. Either supply output_path to Laminator.save_results()"
                    " or set Laminator.output_path."
                )
            else:
                self.output_path = output_path
        if not Path(self.output_path).exists():
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
            print("Created output directory {}".format(self.output_path))
        print("Saving results to {}".format(self.output_path))
        # generate progress bar
        pbar = tqdm(total=5, desc="Saving results")
        # save mode specific results
        self.angles.to_csv(Path(self.output_path, "angles.csv"), index=False)
        pbar.update(1)
        # save general results
        self.intensities.to_csv(Path(self.output_path, "intensities.csv"), index=False)
        pbar.update(1)
        self.avg_intensities.to_csv(Path(self.output_path, "avg_intensities.csv"), index=False)
        pbar.update(1)
        # save distance matrices to .npy files
        np.save(Path(self.output_path, "avg_distances.npy"), self.avg_dist)
        pbar.update(1)
        np.save(Path(self.output_path, "distances.npy"), self.dists)
        pbar.update(1)
        pbar.close()
        print("Results saved to {}".format(self.output_path))
        pbar.close()
        print("Results saved to {}".format(self.output_path))

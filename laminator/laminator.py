from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from numba import jit
from pydiffmap import diffusion_map as dm
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from skimage import io
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.neighbors import kneighbors_graph
from sknetwork.clustering import Louvain
from tqdm import tqdm

# Reimplementation of Laminator code publish with: https://doi.org/10.1038/s41587-023-01747-2


class Laminator:
    """Laminator class for "laminating" images and assessing the orientation of the "laminated" images."""

    def __init__(self, **kwargs) -> None:
        """
        Laminator class for "laminating" images and assessing the orientation of the "laminated" images.

        :param kwargs:
        """
        # get kwargs and initialize other parameters
        self.mask_path = kwargs.get("mask_path", None)
        self.image_paths = kwargs.get("image_paths", None)
        self.image_names = kwargs.get("image_names", None)
        self.output_path = kwargs.get("output_path", None)
        self.images = kwargs.get("images", None)
        self.mask = kwargs.get("mask", None)
        self.dists = None
        self.avg_dist = None
        self.mds_embedding = None
        self.umap_embedding = None
        self.diffusion_map_embedding = None
        self.pca_embedding = None
        self.intensities = None
        self.louvain_labels = None
        self.knn_graph = None
        self._mode = None

    def set_image_names(self, names: list) -> Laminator:
        """
        Set image names

        :param names:
        :return:
        """
        # generate dict from list of image names
        names = {i: name for i, name in enumerate(names)}
        self.image_names = names

    def load_images(self, paths: list) -> Laminator:
        """
        Load images

        :param paths:
        :return:
        """
        # TODO rewrite with np.array axis=-1
        imgs = []
        for path in tqdm(paths, desc="Loading images", unit=" images"):
            img = io.imread(path)
            imgs.append(img)
        self.image_paths = paths.copy()
        self.images = np.dstack(imgs)

    def load_mask(self, arr: np.ndarray = None, path: str = None) -> Laminator:
        """
        Load mask

        :param arr:
        :param path:
        :return:
        """
        if arr is not None:
            self.mask = arr

        elif path is not None:
            self.mask_path = path
            self.mask = io.imread(path)
        else:
            raise ValueError('Either "arr" or "path" must be specified')
        # check if self.mask contains only 0 and 1
        if set(np.unique(self.mask).tolist()) != {0, 1}:
            raise ValueError("Mask contains values other than 0 and 1")

    def average_intensities(self) -> Laminator:
        """
        Averages intensities over all points

        :return:
        """
        if self.intensities is None:
            raise ValueError("No intensities found. Please run calculate_radial_profiles() first.")
        else:
            self.avg_intensities = self.intensities.copy()
            self.avg_intensities.drop(columns=["radial_position"], inplace=True)
            # TODO: unify format of self.avg_intensities and self.intensities regardless of subclass / mode
            if self._mode == "neighborhood":
                self.avg_intensities = self.avg_intensities.groupby(["image", "label", "x", "y"]).mean().reset_index()
            if self._mode == "laminate":
                self.avg_intensities = self.avg_intensities.groupby(["image", "window"]).mean().reset_index()

    # DISTANCES
    @staticmethod
    @jit(nopython=True)
    def pdist_numba(array: np.ndarray, n_c: int = None):
        """
        Calculates the distance between all rows in array using the first n_c fourier coefficients

        :param array:
        :param n_c:
        :return:
        """
        n = array.shape[0]
        dist = np.zeros((n, n), dtype=np.float64)
        if n_c is None:
            n_c = int(np.floor(array.shape[1] / 2) + 1)
        for i in range(n):
            for j in range(n):
                x = array[i, :]
                y = array[j, :]
                fft1 = np.fft.fft(x)
                fft2 = np.fft.fft(y)
                d = np.sqrt(np.sum(np.abs(fft1[:n_c] - fft2[:n_c]) ** 2))
                dist[i, j] = d
        # bring to square form
        dist = dist + dist.T
        return dist

    def calculate_distances(
        self,
        image: int,
        smoothing: bool = True,
        sigma: int = None,
        rescale: bool = True,
        n_components: int = None,
        log_transform: bool = False,
        interpolate: bool = True,
    ) -> np.ndarray:  # TODO super class
        """
        Calculate distances between radial profiles for a given image

        :param image:
        :param smoothing:
        :param sigma:
        :param rescale:
        :param n_components:
        :param log_transform:
        :param interpolate:
        :return:
        """
        df = self.intensities[self.intensities["image"] == image]
        # pivot window wider
        if self._mode == "laminate":
            df_wide = df.pivot(index="radial_position", columns="window", values="intensity")
        else:
            df_wide = df.pivot(index="radial_position", columns="label", values="intensity")
        # convert to numpy array
        arr = df_wide.to_numpy().T
        # transform log1p
        if log_transform:
            arr = np.log1p(arr)
        # scale between 0 and 1
        if rescale:
            np.interp(arr, (np.percentile(arr, 0), np.percentile(arr, 100)), (0, 1))
        # interpolate array to half the size along columns
        if interpolate:
            f = interp1d(np.arange(arr.shape[1]), arr, axis=1)
            arr = f(np.linspace(0, arr.shape[1] - 1, int(arr.shape[1] / 2)))
        # smooth array along columns
        if smoothing:
            arr = gaussian_filter1d(arr, sigma=sigma, axis=0)
        dist = self.pdist_numba(arr, n_components)
        return dist

    def get_distances(self, sigma: int = 1, n_components: int = None) -> Laminator:  # TODO -> super class
        """
        Calculate distances between radial profiles

        :param sigma:
        :param n_components:
        :return:
        """
        self.dists = []

        if self.intensities is None:
            raise ValueError("No intensities found. Run .extract_intensities() first.")
        for image in tqdm(self.intensities.image.unique(), desc="Calculating distances"):
            dist = self.calculate_distances(image=image, sigma=sigma, n_components=n_components)
            self.dists.append(dist)

        self.dists = np.asarray(self.dists)
        self.avg_dist = np.mean(self.dists, axis=0)

    # EMBEDDINGS
    def mds_scaling(self, n: int = 2) -> Laminator:  # super class
        """
        Run MDS on the embedding

        :param n:
        :return:
        """
        if self.avg_dist is None:
            raise ValueError("Run .get_distances() first.")
        mds = MDS(n_components=n, dissimilarity="precomputed")
        self.mds_embedding = mds.fit_transform(self.avg_dist)

    def pca(self, features: str, n: int = 2) -> Laminator:
        """
        Run PCA on the embedding

        :param type:
        :param n:
        :return:
        """
        if features == "avg_intensities":
            if self.avg_intensities is None:
                raise ValueError("Run .avg_intensities() first.")
            # pivot self.avg_intensities wider for image
            embedding = self.avg_intensities.pivot(index="label", columns="image", values="intensity")
        else:
            raise ValueError('Unknown embedding type. Choose from "avg_intensities".')
        pca = PCA(n_components=n)
        self.pca_embedding = pca.fit_transform(embedding)

    def helper_diffusion_map_from_dist(self, x: np.ndarray, y: np.ndarray) -> float:  # super class
        """
        Helper function for diffusion map.

        Returns precomputed averaged distances between two observables x and y

        :param x:
        :param y:
        :return:
        """
        d = self.avg_dist[int(x[0]), int(y[0])]  # select idx from x and y
        return d

    def diffusion_map(
        self, n: int = 10, epsilon: int = 1, alpha: float = 0.5, k: int = 10, detect_epsilon: bool = True
    ) -> Laminator:  # super class
        """
        Run diffusion map on the embedding

        :param n:
        :param epsilon:
        :param alpha:
        :param k:
        :param detect_epsilon:
        :return:
        """
        if self.avg_dist is None:
            raise ValueError("Run .get_distances() first.")
        if detect_epsilon:
            epsilon = "bgh"
        dm_constructor = dm.DiffusionMap.from_sklearn(
            n_evecs=n, epsilon=epsilon, alpha=alpha, k=k, metric=self.helper_diffusion_map_from_dist
        )
        # generate vector of indices for axis 0 of self.avg_dist
        idx = np.arange(self.avg_dist.shape[0])
        # add idx to axis=1, work around for helper function that takes two arguments from array
        idx = np.stack((idx, np.zeros(idx.shape, dtype=int)), axis=1)
        self.diffusion_map_embedding = dm_constructor.fit_transform(idx)

    def umap(self, reduction: str) -> Laminator:  # super class
        """
        Run UMAP on the embedding

        :param reduction:
        :return:
        """
        reducer = umap.UMAP()
        if reduction == "mds":
            self.umap_embedding = reducer.fit_transform(self.mds_embedding)
        if reduction == "diffusion_map":
            self.umap_embedding = reducer.fit_transform(self.diffusion_map_embedding)
        if reduction == "pca":
            self.umap_embedding = reducer.fit_transform(self.pca_embedding)

    def plot_distances(self, feature: int = None) -> Laminator:  # super class
        """
        Plot the distances

        :param feature:
        :return:
        """
        if feature is None:
            io.imshow(self.avg_dist)
            plt.show()
        else:
            io.imshow(self.dists[feature])
            plt.show()
        # TODO: add option for clustermap of distances and color labeling of louvain clusters etc

    def louvain_clustering(self, embedding_type: str = "umap", n: int = 5) -> Laminator:  # super class
        """
        Run louvain clustering on the embedding

        :param n:
        :param embedding_type:
        :return:
        """
        if embedding_type == "umap":
            embedding = self.umap_embedding
        elif embedding_type == "mds":
            embedding = self.mds_embedding
        elif embedding_type == "diffusion_map":
            embedding = self.diffusion_map_embedding
        elif embedding_type == "pca":
            embedding = self.pca_embedding
        else:
            raise ValueError('Unknown embedding type. Choose from "umap", "diffusion_map", "pca" or "mds".')
        self.knn_graph = kneighbors_graph(embedding, n, mode="connectivity", include_self=True)
        louvain = Louvain()
        self.louvain_labels = louvain.fit_predict(self.knn_graph)

    def plot_embedding(self, embedding_type: str, label: str = None) -> None:  # super class
        """
        Plot the embedding

        :param label:
        :param embedding_type:
        :return:
        """
        if embedding_type == "umap":
            embedding = self.umap_embedding
        elif embedding_type == "mds":
            embedding = self.mds_embedding
        elif embedding_type == "diffusion_map":
            embedding = self.diffusion_map_embedding
        elif embedding_type == "pca":
            embedding = self.pca_embedding
        elif embedding_type == "spatial":
            embedding = self.positions[["x", "y"]].to_numpy()
        else:
            raise ValueError('Unknown embedding type. Choose from "umap", "diffusion_map" or "mds".')
        fig, ax = plt.subplots(figsize=(10, 10))
        if label == "louvain":
            if self.louvain_labels is None:
                raise ValueError("Run Laminator.louvain_clustering() first.")
            sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=self.louvain_labels, palette="tab20")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1])
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
        if embedding_type == "spatial":
            # invert y-axis
            ax.invert_yaxis()
            # equal x and y scaling
            ax.set_aspect("equal", "box")
        ax.set_title(f"{embedding_type} embedding")
        plt.show()

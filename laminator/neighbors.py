from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from skimage import io, measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import rotate
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from laminator import Laminator


class LaminarNeighbors(Laminator):
    """Subclass of Laminator for extracting intensities from radial neighborhoods around points in label image"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.msg_passed_intensities = None
        self.dict_pos = None
        self.dict_louvain = None
        self.radius = kwargs.get("radius", None)
        self.label_image = kwargs.get("label_image", None)
        self.positions = kwargs.get("positions", None)
        self.positions_path = kwargs.get("positions_path", None)
        self.avg_intensities = None
        self.label_image_path = kwargs.get("label_image_path", None)
        self.graph = None
        self.adata = None
        self._mode = "neighborhood"

    def load_label_image(self, path: str) -> Laminator:
        """
        Load label image

        :param path:
        :return:
        """
        self.label_image_path = path
        self.label_image = io.imread(path)

    def load_positions(self, path: str, sep: str = ",") -> Laminator:
        """
        Load positions

        :param sep:
        :param path:
        :return:
        """
        self.positions_path = path
        self.positions = pd.read_csv(path, sep=sep)

    def build_graph(self, n_neigh, radius=None):  # TODO: add options for graph building, be flexible with None types...
        """
        Build knn graph from laminator positions

        :param n_neigh:
        :param radius:
        :return:
        """
        positions = self.positions[["x", "y"]].to_numpy()
        if radius is not None:
            # build radius restricted knn graph
            self.graph = radius_neighbors_graph(positions, radius, mode="distance", include_self=False)
        else:
            # build knn graph
            self.graph = kneighbors_graph(positions, n_neigh, mode="distance", include_self=False)
        # convert to NetworkX graph
        self.dict_louvain = {i: self.louvain_labels[i] for i in range(len(self.louvain_labels))}
        self.dict_pos = {
            i: (self.positions.iloc[i]["y"], self.positions.iloc[i]["x"]) for i in range(len(self.positions))
        }
        # add louvain labels as node attributes
        self.graph = nx.from_numpy_array(self.graph.toarray(), create_using=nx.DiGraph)
        nx.set_node_attributes(self.graph, self.dict_louvain, "louvain_label")
        nx.set_node_attributes(self.graph, self.dict_pos, "pos")

    @staticmethod
    def get_angle(v1, v2):
        """
        Calculate angle between two vectors

        :param v1:
        :param v2:
        :return:
        """
        # calculate angle between to vectors with atan
        angle = np.arctan2(v2[1] - v1[1], v2[0] - v1[0])
        # convert to degrees
        angle = np.rad2deg(angle)
        return angle

    def get_angles(self):
        """
        Get angles between neighbors

        :return:
        """
        # get angles between neighbors
        for u, v in self.graph.edges():
            v1 = self.dict_pos[u]
            v2 = self.dict_pos[v]
            angle = self.get_angle(v1, v2)
            self.graph[u][v]["angle"] = angle

    def get_oriented_intensity_profile(
        self, angle: int, x: int, y: int, path_length: int, path_width: int, x_offset: int, y_offset: int
    ):
        """
        Get intensity profile of oriented window

        :param path_width:
        :param path_length:
        :param angle:
        :param x:
        :param y:
        :param x_offset:
        :param y_offset:
        :return:
        """
        # Crop image centered to point with size that keeps circle of window
        imgs_cropped = self.images[
            y - path_length + y_offset : y + path_length, x - path_length + x_offset : x + path_length, ...
        ]
        imgs_padded = np.zeros([2 * path_length, 2 * path_length, self.images.shape[2]])
        imgs_padded[y_offset : imgs_cropped.shape[0] + y_offset, x_offset : imgs_cropped.shape[1] + x_offset, ...] = (
            imgs_cropped
        )

        # Apply rotation to original cropped image
        imgs_rotated = rotate(imgs_padded, angle)[
            path_length:, path_length - path_width : path_length + path_width, ...
        ]
        imgs_rotated = np.average(imgs_rotated, axis=1)

        return imgs_rotated

    def get_path_intensities(self, path_width: int = 4):
        """
        Get intensity profiles along paths

        :param path_width:
        :return:
        """
        # get intensities along path
        for u, v in tqdm(self.graph.edges(), desc="Calculating path intensities"):
            intensities = self.get_oriented_intensity_profile(
                angle=self.graph[u][v]["angle"],
                path_length=int(self.graph[u][v]["weight"]),
                path_width=path_width,
                x=int(self.dict_pos[u][1]),
                y=int(self.dict_pos[u][0]),
                x_offset=0,
                y_offset=0,
            )
            self.graph[u][v]["intensities"] = intensities

    # RADIAL NEIGHBORHOODS
    def get_radial_profile(self, x: int, y: int, label: int, exclude_label: bool = False):
        """
        Get radial profile of image

        :param x:
        :param y:
        :param label:
        :param exclude_label:
        :return:
        """
        # define center of circle
        center = [y, x]
        # get offsets
        x_offset, y_offset = int(center[1]) - self.radius, int(center[0]) - self.radius
        x_offset = np.where(x_offset < 0, np.absolute(x_offset), 0)
        y_offset = np.where(y_offset < 0, np.absolute(y_offset), 0)

        # Crop image centered to point with size that keeps circle of window
        img = self.images[
            int(center[0]) - self.radius + y_offset : int(center[0]) + self.radius,
            int(center[1]) - self.radius + x_offset : int(center[1]) + self.radius,
            ...,
        ]
        mask = self.mask[
            int(center[0]) - self.radius + y_offset : int(center[0]) + self.radius,
            int(center[1]) - self.radius + x_offset : int(center[1]) + self.radius,
        ]
        # get center of img
        center_img = [int(img.shape[0] / 2), int(img.shape[1] / 2)]
        # get radial distance from center
        y, x = np.indices(img[..., 0].shape)
        r = np.sqrt((x - center_img[1]) ** 2 + (y - center_img[0]) ** 2)
        r = r.astype(int)

        mask = mask * np.where(r <= self.radius, 1, 0)
        if exclude_label:
            nucleus_id = self.label_image[int(center[0]), int(center[1])]
            label_image = self.label_image[
                int(center[0]) - self.radius + y_offset : int(center[0]) + self.radius,
                int(center[1]) - self.radius + x_offset : int(center[1]) + self.radius,
            ]
            mask = mask * np.where(label_image == nucleus_id, 0, 1)

        # prepare for bincount
        mask = mask.ravel()
        r = r.ravel()
        r = np.delete(r, np.where(mask == 0))
        radial_profiles = []
        for i in range(img.shape[2]):
            img_tmp = img[..., i].ravel()
            img_tmp = np.delete(img_tmp, np.where(mask == 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                tbin = np.bincount(r, img_tmp)
                nr = np.bincount(r)
                radial_profiles.append(np.nan_to_num(tbin / nr))
        df = pd.DataFrame(radial_profiles).T.assign(x=int(center[1]), y=int(center[0]))
        df["radial_position"] = df.index
        df["label"] = label
        # TODO: adapt label to with suffix indicating neighborhood or nucleus
        return df

    def extract_positions(self):
        """
        Extracts positions from label image and stores them in positions

        :return:
        """
        if self.label_image is None:
            raise ValueError("No label image found. Please run load_label_image() first.")
        self.positions = pd.DataFrame(measure.regionprops_table(self.label_image, properties=("centroid", "label")))
        # rename centroid columns to x and y
        self.positions.rename(columns={"centroid-0": "y", "centroid-1": "x"}, inplace=True)
        # TODO also extract features of nuclei and add label with suffix indicating nucleus features

    def calculate_radial_profiles(self):
        """
        Calculates radial profiles for all points in df

        :return:
        """
        if self.positions is None:
            self.extract_positions()
        self.positions["y"] = np.round(self.positions["y"].values)
        self.positions["x"] = np.round(self.positions["x"].values)

        result = []
        for x, y, label in tqdm(
            zip(self.positions["x"].values, self.positions["y"].values, self.positions["label"].values),
            total=len(self.positions),
            desc="Calculating radial profiles",
        ):
            tmp = self.get_radial_profile(x, y, label)
            result.append(tmp)
        # convert to dataframe
        self.intensities = pd.concat(result)
        # pivot longer for all columns with intensities
        self.intensities = self.intensities.melt(
            id_vars=["x", "y", "radial_position", "label"], var_name="image", value_name="intensity"
        )

        # add image names from self.image_names dict
        if self.image_names is not None:
            self.intensities["image"] = self.intensities["image"].map(self.image_names)

    def overview_neighborhoods(
        self,
        img: np.ndarray = None,
        scale_intensity: bool = True,
        overlay_label: bool = True,
        show_plot: bool = True,
        save_plot: bool = False,
        label: str = None,
        file: str = "overview_neighborhoods",
    ) -> Laminator:
        """
        Plot the angles on the original image

        :param label:
        :param overlay_label:
        :param scale_intensity:
        :param img:
        :param show_plot:
        :param save_plot:
        :param file:
        :return:
        """
        if self.positions is None:
            raise ValueError("No positions found. Please run extract_positions() first.")
        # Plot contour on original image
        fig, ax = plt.subplots(figsize=(20, 20))
        if img is not None:
            if scale_intensity:
                img = rescale_intensity(img)
            if overlay_label:
                if self.label_image is None:
                    raise ValueError("No label image found. Please run load_label_image() first.")
                else:
                    label_img = label2rgb(self.label_image, image=img, bg_label=0, alpha=0.5)
                    ax.imshow(label_img)
            else:
                ax.imshow(img, cmap=plt.cm.gray)
        else:
            print("No image provided. Plotting only positions.")
            ax.scatter(self.positions["x"].values, self.positions["y"].values, c="b")
            # invert y-axis
            ax.invert_yaxis()
        # plot circles around each point in self.positions with radius self.radius if labels is louvain
        # color labels by self.louvain_labels
        if label is None:
            for x, y in zip(self.positions["x"].values, self.positions["y"].values):
                c = plt.Circle((x, y), self.radius, color="r", fill=False, alpha=0.5)
                ax.add_patch(c)
        elif label == "louvain":
            for x, y, l in zip(self.positions["x"].values, self.positions["y"].values, self.louvain_labels):
                c = plt.Circle((x, y), self.radius, color=plt.cm.tab20(l), fill=False)
                ax.add_patch(c)

        if save_plot:
            plt.savefig(str(file + ".png"))
        if show_plot:
            plt.show()
        plt.close("all")

    def extract_intensities(self) -> Laminator:  # write wrapper for each subclass
        """
        Extract intensities from images

        :return:
        """
        if self.images is None:
            raise ValueError("No images loaded. Run Laminator.load_images() first.")
        elif self.mask is None:
            self.mask = np.ones(self.images[..., 0].shape, dtype=bool)
        if self.positions is None:
            if self.label_image is None:
                raise ValueError(
                    "No positions or label image loaded."
                    "Run Laminator.load_positions() first or supply label image with Laminator.load_label_image()."
                )
            else:
                self.extract_positions()
        self.calculate_radial_profiles()
        self.average_intensities()

    def construct_features(self, log1p_transform: bool = True, scale=True) -> Laminator:
        """
        Construct nxnxm weighted adjacency matrix. Edges are populated by averaged intensity profiles of edges.

        Self-directed edges are populated by averaged intensity profiles of nuclei.
        :return:
        """
        A = nx.adjacency_matrix(self.graph, weight=None)
        # add self connections on diagonal
        A = A + csr_array(np.diag(np.ones(A.shape[0])))
        # get inverse degree matrix
        D_inv = csr_array(np.linalg.inv(np.diag(np.sum(A, axis=1))))
        # TODO: check for performance!!
        # weight A with D_inv so that all row sums are 1
        A_weight = D_inv.dot(A)
        # initialize nxm feature matrix from self.avg_intensities by reshaping df to wide
        self.msg_passed_intensities = self.avg_intensities.pivot(
            index="label", columns="image", values="intensity"
        ).to_numpy()

        # TODO: reconsider when to log1p transform...
        # log1p
        if log1p_transform:
            self.msg_passed_intensities = np.log1p(self.msg_passed_intensities)
        if scale:
            scaler = StandardScaler()
            scaler.fit(self.msg_passed_intensities)
            self.msg_passed_intensities = scaler.transform(self.msg_passed_intensities)

        # initialize sparse csr feature matrix using scipy.sparse.csr_matrix with shape Axm
        for i in tqdm(range(self.msg_passed_intensities.shape[1]), desc="Calculating message passed intensities"):
            F = csr_array(A.shape)
            # self connections
            F = F + np.diag(self.msg_passed_intensities[:, i])
            if log1p_transform:
                # edge connections
                for u, v in self.graph.edges():
                    F[u, v] = np.log1p(np.mean(self.graph[u][v]["intensities"][:, i]))
            else:
                # edge connections
                for u, v in self.graph.edges():
                    F[u, v] = np.mean(self.graph[u][v]["intensities"][:, i])
            # weight with D_inv
            F = F * A_weight
            # dot product with unit vector
            F = F.dot(np.ones(A.shape[0]))
            # assign features to self.msg_passed_intensities
            self.msg_passed_intensities[:, i] = F

    def pca(self, features: str, n: int = 2) -> Laminator:
        """
        Run PCA on the embedding

        :param features:
        :param n:
        :return:
        """
        pca = PCA(n_components=n)
        if features == "avg_intensities":
            if self.avg_intensities is None:
                raise ValueError("Run .avg_intensities() first.")
            # pivot self.avg_intensities wider for image
            self.pca_embedding = pca.fit_transform(
                self.avg_intensities.pivot(index="label", columns="image", values="intensity")
            )
        if features == "message_passed_intensities":
            if self.msg_passed_intensities is None:
                raise ValueError("Run .construct_features() first.")
            # pivot self.msg_passed_intensities wider for image
            self.pca_embedding = pca.fit_transform(self.msg_passed_intensities)

    def convert_to_anndata(self) -> ad.AnnData:
        """
        Export results to AnnData object

        :return:
        """
        if self.average_intensities is None:
            raise ValueError("No average intensities calculated. Run Laminator.extract_intensities() first.")
        # initialize dataframes
        df = self.avg_intensities.copy()
        df_intensities = self.intensities.copy()

        if self.positions is None:
            raise ValueError("No positions calculated. Run Laminator.extract_intensities() first.")
        df = df.pivot(index="label", columns="image", values="intensity")
        df_meta = self.positions.set_index("label")
        df_meta.index = df_meta.index.astype(str)
        df_intensities = df_intensities.pivot(index=["label", "image"], columns="radial_position", values="intensity")

        # create AnnData object
        self.adata = ad.AnnData(X=df.to_numpy())

        self.adata.obs = df_meta
        self.adata.obs_names = df.index.astype(str)
        self.adata.var_names = df.columns.astype(str)

        # add self.dists as distance matrices
        if self.image_names is None:
            print("No image names specified. Setting generic default names.")
            self.image_names = self.avg_intensities["image"].unique().tolist()
            self.set_image_names(self.image_names)
        # save image related distances and intensity profiles
        for i, image_name in self.image_names.items():
            if self.dists is not None:
                self.adata.obsm[f"distances_{image_name}"] = self.dists[i]
            self.adata.obsm[f"intensity_profiles_{image_name}"] = df_intensities[
                df_intensities.index.get_level_values(1) == image_name
            ].to_numpy()
        # add self.avg_dist as distance matrix
        if self.avg_dist is not None:
            self.adata.obsm["avg_distances"] = self.avg_dist
        # add umap coordinates
        if self.umap_embedding is not None:
            self.adata.obsm["umap"] = self.umap_embedding
        # add pca coordinates
        if self.pca_embedding is not None:
            self.adata.obsm["pca"] = self.pca_embedding
        # add mds coordinates
        if self.mds_embedding is not None:
            self.adata.obsm["mds"] = self.mds_embedding
        # add louvain labels to obs
        if self.louvain_labels is not None:
            self.adata.obs["louvain"] = self.louvain_labels
        # add knn_graph to uns
        if self.knn_graph is not None:
            self.adata.uns["knn_graph"] = self.knn_graph
        # add diffusion map to obsm
        if self.diffusion_map_embedding is not None:
            self.adata.obsm["diffusion_map"] = self.diffusion_map_embedding
        # add graph to uns
        if self.graph is not None:
            self.adata.uns["graph"] = self.graph
        # add msg_passed_intensities as layer
        if self.msg_passed_intensities is not None:
            self.adata.layers["msg_passed_intensities"] = self.msg_passed_intensities

    def export_anndata(self, output_path: str = None) -> None:
        """
        Export results to AnnData object

        :param output_path:
        :return:
        """
        if self.adata is None:
            self.convert_to_anndata()
        if output_path is None:
            output_path = self.output_path
        if output_path is None:
            raise ValueError("No output path specified.")
        else:
            # remove self.adata.uns['intensities'] to save space
            if "intensities" in self.adata.uns:
                del self.adata.uns["intensities"]
            if not Path(output_path).exists():
                Path(output_path).mkdir(parents=True)
                print("Created output directory {}".format(output_path))
            self.adata.write(Path(output_path, "adata.h5ad"))
            # reset self.adata.uns['intensities']
            if self.intensities is not None:
                self.adata.uns["intensities"] = self.intensities
            print("AnnData object saved to {}".format(Path(output_path, "adata.h5ad")))
            if self.intensities is not None:
                self.adata.uns["intensities"] = self.intensities
            print("AnnData object saved to {}".format(Path(output_path, "adata.h5ad")))

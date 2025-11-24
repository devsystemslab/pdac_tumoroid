import anndata as ad
import joblib
import matplotlib.pyplot as plt
import muon as mu
import networkx as nx
import numpy as np
import pymeshfix
from matplotlib.colors import LightSource
from scipy.spatial import ConvexHull
from skimage.measure import marching_cubes
from sklearn.neighbors import radius_neighbors_graph
from tqdm import tqdm
from trimesh import Trimesh, util
from trimesh.smoothing import filter_taubin


def filter_adata(
    adata: ad.AnnData, well: str, plate: str, clusters: list
) -> ad.AnnData:
    adata = adata[adata.obs["plate_id"] == plate]
    adata = adata[adata.obs["well_id"] == well]
    adata = adata[adata.obs["leiden"].isin(clusters)]
    return adata


def get_connected_components(
    adata: ad.AnnData, radius: int = 100, min_nds=10, min_degree=3
) -> nx.Graph:
    pts = adata.obs[["centroid-1", "centroid-0", "z"]].to_numpy()
    # neighbor graph
    G = nx.from_numpy_array(
        radius_neighbors_graph(
            pts, radius, mode="distance", include_self=False
        ).toarray(),
        create_using=nx.DiGraph,
    ).to_undirected()
    # add pts as node attributes
    for i, node in enumerate(G.nodes):
        G.nodes[node]["pos"] = pts[i]

    # filter out points that have less than 3 connections
    for node in list(G.nodes):
        if G.degree[node] < min_degree:
            G.remove_node(node)
    if len(G.nodes) <= min_nds:
        return None
    for i, component in enumerate(list(nx.connected_components(G))):
        if len(component) < min_nds:
            for node in component:
                G.remove_node(node)
        else:
            for node in component:
                G.nodes[node]["component"] = i
    return G


def plot_connected_components_2d(G):
    """
    Plot connected components in 2D
    :param G:
    :return:
    """
    plt.figure(figsize=(10, 10))
    components = [G.nodes[node].get("component") for node in G.nodes]
    coords = {node: G.nodes[node]["pos"][0:2] for node in G.nodes}
    # use pts as coordinate layout
    nx.draw(
        G,
        pos=coords,
        node_size=10,
        node_color=components,
        cmap="tab20",
        edge_color="k",
        alpha=0.5,
        arrows=False,
    )
    # flip y_axis
    plt.gca().invert_yaxis()
    plt.show()


# TODO: expand for duct and fibro graphs, camera position, background color
def plot_chulls_3d(
    G: nx.Graph,
    G_ecm: nx.Graph | None = None,
    n_taubin: int = 10,
    add_chulls: bool = True,
    lambda_taubin: float = 0.5,
    nu_taubin: float = 0.5,
    add_nodes: bool = False,
    add_edges: bool = False,
    add_ecm_nodes: bool = True,
    add_ecm_edges: bool = True,
    format_axes: bool = False,
    return_mesh: bool = False,
    view: tuple[int, int] = (15, 15),
):
    """
    Plot convex hulls of connected components in 3D
    :param G_ecm:
    :param G:
    :param add_chulls:
    :param n_taubin:
    :param lambda_taubin:
    :param nu_taubin:
    :param add_nodes:
    :param add_edges:
    :param add_ecm_nodes:
    :param add_ecm_edges:
    :param format_axes:
    :param return_mesh:
    :param view:
    :return:
    """

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(view[0], view[1])

    nodes = np.array([G.nodes[v]["pos"] for v in sorted(G)])
    edges = np.array([(G.nodes[u]["pos"], G.nodes[v]["pos"]) for u, v in G.edges()])

    if add_nodes:
        ax.scatter(*nodes.T, s=20, c="black", ec="w")
    if add_edges:
        for vizedge in edges:
            ax.plot(*vizedge.T, color="tab:gray")
    if add_chulls:
        # for each component get trimesh
        mesh = []
        for i, component in enumerate(nx.connected_components(G)):
            # get nodes
            nodes = np.array([G.nodes[v]["pos"] for v in component])
            # get convex hull
            hull = ConvexHull(nodes)
            # get trimesh
            trimesh = Trimesh(vertices=nodes, faces=hull.simplices)
            # pymeshfix
            trimesh.vertices, trimesh.faces = pymeshfix.clean_from_arrays(
                trimesh.vertices, trimesh.faces
            )
            # smooth
            trimesh = filter_taubin(
                trimesh, iterations=n_taubin, lamb=lambda_taubin, nu=nu_taubin
            )
            mesh.append(trimesh)
            # plot
            ax.plot_trisurf(
                trimesh.vertices[:, 0],
                trimesh.vertices[:, 1],
                triangles=trimesh.faces,
                Z=trimesh.vertices[:, 2],
                alpha=0.75,
            )
        mesh = util.concatenate(mesh)

    if G_ecm is not None:
        nodes_ecm = np.array([G_ecm.nodes[v]["pos"] for v in sorted(G_ecm)])
        edges_ecm = np.array(
            [(G_ecm.nodes[u]["pos"], G_ecm.nodes[v]["pos"]) for u, v in G_ecm.edges()]
        )
        if add_ecm_nodes:
            ax.scatter(*nodes_ecm.T, s=20, c="black", ec="w")
        if add_ecm_edges:
            for vizedge in edges_ecm:
                ax.plot(*vizedge.T, color="tab:gray")

    ax.set_box_aspect([1, 1, 1])

    if format_axes:
        _format_axes(ax)
    fig.tight_layout()
    plt.show()
    if return_mesh and add_chulls:
        return mesh


def create_surface_plot(lumen_mask_label, organoid_mask, file_name, pad=0):
    # Setup matplotlib
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, wspace=0)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])
    max_size = (np.array(lumen_mask_label.shape) * 4 * 0.347).max() + pad
    ax.set_xlim3d([0, max_size])
    ax.set_ylim3d([0, max_size])
    ax.set_zlim3d([0, max_size])
    shapes = lumen_mask_label.shape
    ax.view_init(15, 15)
    ax.tick_params(axis="both", pad=5)
    ls = LightSource(azdeg=0, altdeg=-65)

    # Loop over each lumen
    for label in tqdm(range(1, len(np.unique(lumen_mask_label)))):
        # Extract surfaces and cleanup vertices and faces
        vertices, faces, _, _ = marching_cubes(
            (lumen_mask_label == label).astype(int), 0, step_size=1
        )
        vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)
        vertices_clean = np.round(vertices_clean * 4 * 0.347).astype(int)
        # Convert to trimesh and smooth
        cell_mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)
        cell_mesh = filter_taubin(cell_mesh, iterations=50)

        # Create trisurf plot
        ax.plot_trisurf(
            cell_mesh.vertices[:, 0],
            cell_mesh.vertices[:, 1],
            triangles=cell_mesh.faces,
            Z=cell_mesh.vertices[:, 2],
            lightsource=ls,
            alpha=1.0,
        )

    # Do the same for whole organoid in gray
    vertices, faces, _, _ = marching_cubes(
        (organoid_mask == 1).astype(int), 0, step_size=2
    )
    vertices_clean, faces_clean = pymeshfix.clean_from_arrays(vertices, faces)
    vertices_clean = np.round(vertices_clean * 4 * 0.347).astype(int)

    cell_mesh = Trimesh(vertices=vertices_clean, faces=faces_clean)
    cell_mesh = filter_taubin(cell_mesh, iterations=50)

    # Create trisurf plot
    ax.plot_trisurf(
        cell_mesh.vertices[:, 0],
        cell_mesh.vertices[:, 1],
        triangles=cell_mesh.faces,
        Z=cell_mesh.vertices[:, 2],
        lightsource=ls,
        alpha=0.3,
        cmap="gray",
    )

    # Cleanup axis
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    xs = [0, 200, 400, 600]
    ax.set_xticks(xs)

    ax.set_xticklabels(ax.get_xticks(), va="bottom")

    # Save image file
    plt.savefig(file_name, pad_inches=0, dpi=300)
    plt.close()


def process_well(
    well,
    plate,
    mdata,
    cycle,
    mod="phenocoder",
    min_degree_ducts=5,
    min_degree_fibro=1,
    min_nds_ducts=10,
    mind_nds_fibro=10,
    clusters_ducts=["3", "4"],
    clusters_fibro=["0", "1", "2"],
):
    """
    Process well
    :param well:
    :param plate:
    :param mdata:
    :param mod:
    :param min_degree_ducts:
    :param min_degree_fibro:
    :param min_nds_ducts:

    :param clusters_ducts:
    :param clusters_fibro:
    :return:
    """
    # select sample
    adata_ducts = filter_adata(mdata[mod], well, plate, clusters=clusters_ducts)
    adata_fibro = filter_adata(mdata[mod], well, plate, clusters=clusters_fibro)
    # get connected components and graph
    graph_ducts = get_connected_components(
        adata_ducts, radius=100, min_nds=min_nds_ducts, min_degree=min_degree_ducts
    )
    graph_fibro = get_connected_components(
        adata_fibro, radius=200, min_nds=mind_nds_fibro, min_degree=min_degree_fibro
    )
    # plot 2d
    plot_connected_components_2d(graph_ducts)
    plot_connected_components_2d(graph_fibro)
    # plot chulls 3d
    mesh = plot_chulls_3d(
        graph_ducts,
        graph_fibro,
        n_taubin=1,
        nu_taubin=0.5,
        lambda_taubin=0.5,
        add_ecm_nodes=False,
        add_ecm_edges=False,
        view=(270, 0),
        format_axes=True,
        return_mesh=True,
    )
    # save to ply file
    mesh.export(f"{dir_output}/ducts_{plate}-{cycle}-{well}.ply")
    # save graphs with joblib
    joblib.dump(graph_ducts, f"{dir_output}/graph_ducts-{plate}-{cycle}-{well}.joblib")
    joblib.dump(graph_fibro, f"{dir_output}/graph_fibro-{plate}-{cycle}-{well}.joblib")


if __name__ == "__main__":
    # set paths
    dir_adata = "data/processed/inhibitors/anndata"
    dir_output = "data/processed/inhibitors/graph_segmentations"
    # set files
    file_cycle_01 = f"{dir_adata}/mdata_cycle-01.h5mu"
    file_cycle_03 = f"{dir_adata}/mdata_cycle-03.h5mu"
    # read mdatas
    mdata_cycle_01 = mu.read_h5mu(file_cycle_01)
    mdata_cycle_03 = mu.read_h5mu(file_cycle_03)
    # process well
    well = "J08"
    plate = "004"
    process_well(
        well,
        plate,
        mdata_cycle_01,
        "01",
        min_nds_ducts=5,
        clusters_ducts=["2", "3"],
        clusters_fibro=["1", "4", "5", "0"],
    )
    process_well(well, plate, mdata_cycle_03, "03", min_nds_ducts=5, min_degree_ducts=4)

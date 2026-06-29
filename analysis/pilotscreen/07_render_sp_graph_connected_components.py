import os

import joblib
import networkx as nx
import numpy as np

# # Set environment variables for Apple Silicon compatibility
# os.environ['SYSTEM_VERSION_COMPAT'] = '0'
# os.environ['ARCHFLAGS'] = '-arch arm64'
# # Also keep the native window setting
# os.environ['OPEN3D_USE_NATIVE_WINDOW'] = '1'
import open3d as o3d
import open3d.visualization.gui as gui


def render_ducts_and_fibroblast_network(file_ducts: str, file_graph_fibro: str):
    """
    Render ducts and fibroblast network
    :param file_ducts:
    :param file_graph_fibro:
    :return:
    """

    mesh = o3d.io.read_triangle_mesh(file_ducts)
    mesh.compute_vertex_normals()
    line_set_ducts = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    mat_duct = o3d.visualization.rendering.MaterialRecord()
    mat_duct.shader = "unlitLine"
    mat_duct.line_width = 4
    mat_duct.base_color = [1, 0, 1, 0.5]
    mat_duct.base_roughness = 0.0
    mat_duct.base_reflectance = 0.0
    mat_duct.base_clearcoat = 1.0
    mat_duct.thickness = 2.0
    mat_duct.transmission = 1.0
    mat_duct.absorption_distance = 10
    mat_duct.absorption_color = [0.5, 0.5, 0.5]

    mat_sphere = o3d.visualization.rendering.MaterialRecord()
    mat_sphere.shader = "defaultLitTransparency"
    mat_sphere.base_color = [1, 1, 1, 0.75]

    graph_fibro = joblib.load(file_graph_fibro)
    graph_fibro = nx.convert_node_labels_to_integers(graph_fibro, first_label=0)
    coords = nx.get_node_attributes(graph_fibro, "pos")
    coords = [coords[i].tolist() for i in range(len(coords))]
    edges = list(graph_fibro.edges)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(coords),
        lines=o3d.utility.Vector2iVector(edges),
    )
    # line_set.paint_uniform_color([0.3, 0.3, 0.3])

    mat_box = o3d.visualization.rendering.MaterialRecord()
    # mat_box.shader = 'defaultLitTransparency'
    # mat_box.shader = 'defaultLitSSR'
    # mat_box.shader = 'unlitLine'
    mat_box.line_width = 2
    mat_box.base_color = [0.5, 0.5, 0.5, 0.5]
    # mat_box.base_roughness = 0.0
    # mat_box.base_reflectance = 0.0
    # mat_box.base_clearcoat = 1.0
    # mat_box.thickness = 2.0
    # mat_box.transmission = 1.0
    # mat_box.absorption_distance = 10
    # mat_box.absorption_color = [0.5, 0.5, 0.5]

    geoms = [
        {"name": "lines", "geometry": line_set, "material": mat_box},
        {"name": "mesh", "geometry": mesh, "material": mat_sphere},
        {"name": "mesh_lines", "geometry": line_set_ducts, "material": mat_duct},
    ]
    o3d.visualization.draw(geoms, show_skybox=False, show_ui=True)


if __name__ == "__main__":
    dir_files = "whole_mount_tumoroid/analysis/inhibitors/graph_segmentations"
    file_ducts = os.path.join(dir_files, "ducts_004-03-J08.ply")
    file_fibro = os.path.join(dir_files, "graph_fibro-004-03-J08.joblib")
    render_ducts_and_fibroblast_network(file_ducts, file_fibro)

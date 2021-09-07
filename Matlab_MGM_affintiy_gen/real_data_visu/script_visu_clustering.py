import os
import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import numpy as np
import networkx as nx
import argparse


def get_visb_sc_shape(visb_sc):
    """
    get the subplot shape in a visbrain scene
    :param visb_sc:
    :return: tuple (number of rows, number of coloumns)
    """
    k = list(visb_sc._grid_desc.keys())
    return k[-1]



def visbrain_plot(mesh, tex=None, caption=None, cblabel=None, visb_sc=None,
                  cmap='jet'):
    """
    Visualize a trimesh object using visbrain core plotting tool
    :param mesh: trimesh object
    :param tex: numpy array of a texture to be visualized on the mesh
    :return:
    """
    from visbrain.objects import BrainObj, ColorbarObj, SceneObj
    b_obj = BrainObj('gui', vertices=np.array(mesh.vertices),
                     faces=np.array(mesh.faces),
                     translucent=False,
                     hemisphere="both")
    #b_obj.rotate(fixed="bottom", scale_factor=0.02)
    if visb_sc is None:
        visb_sc = SceneObj(bgcolor='white', size=(1400, 1000))
        visb_sc.add_to_subplot(b_obj, title=caption)
        visb_sc_shape = (1, 1)
    else:
        visb_sc_shape = get_visb_sc_shape(visb_sc)
        visb_sc.add_to_subplot(b_obj, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1], title=caption)

    if tex is not None:
        b_obj.add_activation(data=tex, cmap=cmap,
                             clim=(np.min(tex), np.max(tex)))
        CBAR_STATE = dict(cbtxtsz=20, txtsz=20., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), cblabel=cblabel)
        cbar = ColorbarObj(b_obj, **CBAR_STATE)
        visb_sc.add_to_subplot(cbar, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1] + 1, width_max=200)
    return visb_sc


def get_clustering_tex(path_to_folder, size_mesh):
    """
    Get the texture to visualise the clustering of pits on the mesh.
    Return two textures one with the points with labels and one with
    the point without labels
    """

    # Initialise the result texture
    texture_clustered = np.zeros(size_mesh)
    texture_non_clustered = np.zeros(size_mesh)

    # go through the graphs
    graph_folder_path = os.path.join(path_to_folder, "modified_graphs")
    for graph_file_name in os.listdir(graph_folder_path):

        graph_path = os.path.join(graph_folder_path, graph_file_name)
        graph = nx.read_gpickle(graph_path)

        # Go through all the nodes of the graph
        for node in graph.nodes:
            if not graph.nodes[node]["is_dummy"]:
                if graph.nodes[node]["label_dbscan"] != -1:
                    texture_clustered[graph.nodes[node]["ico100_7_vertex_index"]] = graph.nodes[node]["label_dbscan"] + 1
                else:
                    texture_non_clustered[graph.nodes[node]["ico100_7_vertex_index"]] = 1

    return texture_clustered, texture_non_clustered


def read_graphs_and_start_preview(path_to_folder, mesh_path):
    """
    Load the graphs and mesh and create the right texture for the mesh
    where each colored points is a node in a graph where the color indicate the cluster
    it belongs to.
    """
    # get the mesh
    mesh_clustering = sio.load_mesh(mesh_path)

    # transform the mesh
    transl = [50, 50, 0]
    transfo_lh = np.array([[-1, 0, 0, transl[0]],[0, -1, 0, transl[1]],[0, 0, 1, transl[2]], [0,0,0,50]])
    transfo_full = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    mesh_clustering.apply_transform(transfo_full)

    print(mesh_clustering)

    # get the texture given the clustering of the graph
    total_nb_vertices = mesh_clustering.vertices.shape[0]
    texture_clustered, texture_non_clustered = get_clustering_tex(path_to_folder, total_nb_vertices)

    # plot the mesh to check that it works
    visb_sc = visbrain_plot(mesh=mesh_clustering,
                            tex=texture_clustered,
                            caption='Clustered texture',
                            cmap="jet")

    visb_sc = visbrain_plot(mesh=mesh_clustering,
                            tex=texture_non_clustered,
                            caption='Clustered texture',
                            cmap="jet",
                            visb_sc=visb_sc)
    

    visb_sc.preview()


if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Visualise the clustering result")
    parser.add_argument("path_to_folder", help="path to a folder which contains the graphs")
    parser.add_argument("--path_mesh",
                        help="path to the mesh file",
                        default="/Users/Nathan/stageINT/stage_nathan/data_pits_graph/template/lh.OASIS_testGrp_average.gii")
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    path_mesh = args.path_mesh

    read_graphs_and_start_preview(path_to_folder, path_mesh)
    

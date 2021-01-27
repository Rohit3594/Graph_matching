import os
import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import slam.differential_geometry as sdg
import numpy as np
import networkx as nx
import argparse
import pickle



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


def load_graphs_in_list(path):
    """
    Return a list of graph loaded from the path
    """
    path_to_graphs_folder = os.path.join(path, "modified_graphs")
    list_graphs = []
    for i_graph in range(0,len(os.listdir(path_to_graphs_folder))):
        path_graph = os.path.join(path_to_graphs_folder, "graph_"+str(i_graph)+".gpickle")
        graph = nx.read_gpickle(path_graph)
        list_graphs.append(graph)
        
    return list_graphs


def create_clusters_lists(list_graphs):
    """
    Given a list of graphs, return a list of list that represents the clusters.
    Each inside list represent one cluster and each elemnt of the cluster is
    a tuple (graph_number, node_in_the_graph).
    """
    
    result_dict = {}
    
    for i_graph, graph in enumerate(list_graphs):
        for node in graph.nodes:
            
            label_cluster = graph.nodes[node]["label_dbscan"]
            if label_cluster != -1 and not graph.nodes[node]["is_dummy"]:
                if label_cluster in result_dict:
                    result_dict[label_cluster].append((i_graph, node))
                else:
                    result_dict[label_cluster] = [(i_graph, node)]
                    
    # We make sure that every clusters have more than one element              
    return {i: result_dict[i] for i in result_dict if len(result_dict[i]) > 1}


def get_one_point_smoothed_texture(node_information, mesh, nb_iter=10, dt=0.5):
    """
    Return the smoothed texture (using laplacian smoothing) of one sulcal pits point.
    """

    mesh_size = mesh.vertices.shape[0]

    # calculate the non smoothed texture
    non_smoothed_texture = np.zeros(mesh_size)
    non_smoothed_texture[node_information["ico100_7_vertex_index"]] = 1

    # smooth the texture
    smoothed_texture = sdg.laplacian_texture_smoothing(mesh,
                                                       non_smoothed_texture,
                                                       nb_iter,
                                                       dt)

    return smoothed_texture

def get_one_cluster_smoothed_texture(cluster_num, list_graphs, cluster_dict, mesh, nb_iter, dt):
    """
    Return the smoothed texture (using laplacian smoothing) of one cluster
    of sulcal pits.
    """

    mesh_size = mesh.vertices.shape[0]

    # calculate the non smoothed texture
    non_smoothed_texture = np.zeros(mesh_size)

    for graph, node in cluster_dict[cluster_num]:
        vertex_position = list_graphs[graph].nodes[node]["ico100_7_vertex_index"]
        non_smoothed_texture[vertex_position] = 1

    
    # smooth the texture
    smoothed_texture = sdg.laplacian_texture_smoothing(mesh,
                                                       non_smoothed_texture,
                                                       nb_iter,
                                                       dt)

    return smoothed_texture

def get_non_clustered_points_smoothed_texture(list_graphs, mesh, nb_iter, dt):
    """
    Return the smoothed texture of all non labeled points
    """

    mesh_size = mesh.vertices.shape[0]

    # initialise texture
    non_smoothed_texture = np.zeros(mesh_size)

    for graph in list_graphs:
        for node in graph.nodes:
            if not graph.nodes[node]["is_dummy"] and graph.nodes[node]["label_dbscan"] == -1:
                non_smoothed_texture[graph.nodes[node]["ico100_7_vertex_index"]] = 1

    # smooth the texture
    smoothed_texture = sdg.laplacian_texture_smoothing(mesh,
                                                       non_smoothed_texture,
                                                       nb_iter,
                                                       dt)

    return smoothed_texture


def calculate_and_show_pits_smoothed_clusters(path_to_graphs,
                                              path_mesh,
                                              path_smoothed_clustered_texture,
                                              path_to_save):
    """
    Automate the calculation of the smoothed texture for
    sulcal pits clusters and start a preview.
    """

    # Get the graphs and their clusters
    list_graphs = load_graphs_in_list(path_to_graphs)
    cluster_dict = create_clusters_lists(list_graphs)

    # Get the mesh
    mesh_clustering = sio.load_mesh(path_mesh)

    transfo_full = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    mesh_clustering.apply_transform(transfo_full)
    
    mesh_size = mesh_clustering.vertices.shape[0]

    # Get the texture for each cluster
    nb_iter = 100
    dt=0.08

    # Calculate or load the texture for the labeled points.
    if path_smoothed_clustered_texture == "":
        
        # initialize texture for each cluster
        mat_texture = np.zeros((len(cluster_dict), mesh_size))

        # Calculate the texture of each cluster
        for cluster_i, cluster_num in enumerate(cluster_dict.keys()):

            texture = get_one_cluster_smoothed_texture(cluster_num,
                                                       list_graphs,
                                                       cluster_dict,
                                                       mesh_clustering,
                                                       nb_iter,
                                                       dt)
            mat_texture[cluster_i,:] = texture

        # Take the max value for each vertex
        final_clustered_texture = mat_texture.max(0)

        # save if necessary
        if path_to_save != "":
            np.save(path_to_save, final_clustered_texture)

    else:
        final_clustered_texture = np.load(path_smoothed_clustered_texture)

    
    # Calculate or load the texture for the unlabeled points

    final_non_clustered_texture = get_non_clustered_points_smoothed_texture(list_graphs,
                                                                                mesh_clustering,
                                                                                nb_iter,
                                                                                dt)

    # preview the two textures

    visb_sc = visbrain_plot(mesh=mesh_clustering,
                            tex=final_clustered_texture,
                            caption='Test du mesh',
                            cmap="jet")
    
    visb_sc = visbrain_plot(mesh=mesh_clustering,
                            tex=final_non_clustered_texture,
                            caption='Test du mesh',
                            cmap="jet",
                            visb_sc=visb_sc)

        
    visb_sc.preview()



if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Visualise the clustering result with density smoothing")
    parser.add_argument("path_to_folder",
                        help="path to a folder which contains the graphs")
    parser.add_argument("--path_mesh",
                        help="path to the mesh file",
                        default="/Users/Nathan/stageINT/stage_nathan/data_pits_graph/template/lh.OASIS_testGrp_average.gii")
    parser.add_argument("--path_smoothed_texture",
                        help="Path to the file with precalculated smoothed clustered texture",
                        default="")
    parser.add_argument("--path_to_save",
                        help="Path to save the clustered texture once it is calculated. If not prrovided, do not save the file",
                        default="")
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    path_mesh = args.path_mesh
    path_smoothed_texture = args.path_smoothed_texture
    path_to_save=args.path_to_save

    calculate_and_show_pits_smoothed_clusters(path_to_folder,
                                              path_mesh,
                                              path_smoothed_texture,
                                              path_to_save)

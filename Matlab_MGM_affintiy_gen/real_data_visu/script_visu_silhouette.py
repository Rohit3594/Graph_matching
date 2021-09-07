import os
import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import slam.differential_geometry as sdg
import slam.generate_parametric_surfaces as sps
from visbrain.objects import VispyObj, SourceObj
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


def get_centroid_clusters(list_graphs, clusters_dict):
    """
    Return a dictionary which gives for each cluster the belonging point
    which is the closest to the centroid
    """

    result_dict = {}

    for cluster_key in clusters_dict:

        # initialise the matrix which holds the position of all the point in the cluster
        position_mat = np.zeros((len(clusters_dict[cluster_key]), 3))

        # fill the matrix
        for elem_i, (graph_num, node) in enumerate(clusters_dict[cluster_key]):

            graph = list_graphs[graph_num]
            position_mat[elem_i,:] = graph.nodes[node]["sphere_3dcoords"]

        # get the centroid
        centroid = position_mat.mean(0)

        # get the closest point to the centroid
        min_distance = -1
        for graph_num, node in clusters_dict[cluster_key]:

            graph = list_graphs[graph_num]
            position_node = graph.nodes[node]["sphere_3dcoords"]
            distance_to_centroid = np.linalg.norm(centroid - position_node)

            if distance_to_centroid < min_distance or min_distance == -1:
                min_distance = distance_to_centroid
                point_to_save = (graph_num, node)

        result_dict[cluster_key] = point_to_save

    return result_dict



def get_all_silhouette_value(list_graphs, cluster_dict):
    """
    Return a dict with all the silhouette value that can be calculated for each cluster
    """
    
    result_dict = {}
    
    for cluster_key in cluster_dict:
        for main_counter in range(len(cluster_dict[cluster_key])):
            
            graph_main, node_main = cluster_dict[cluster_key][main_counter]
            vector_1 = list_graphs[graph_main].nodes[node_main]["sphere_3dcoords"]
            
            # We calculate the inter cluster similarity
            a_list = []
            for inter_cluster_counter in range(len(cluster_dict[cluster_key])):
                
                if main_counter != inter_cluster_counter:
                    graph_inter, node_inter = cluster_dict[cluster_key][inter_cluster_counter]
                    vector_2 = list_graphs[graph_inter].nodes[node_inter]["sphere_3dcoords"]
                
                    distance = np.linalg.norm(vector_1 - vector_2)
                    a_list.append(distance)
                
            a = np.mean(a_list)
            
            
            # We calculate the average distance with points from other cluster
            b_list = []
            for cluster_intra_key in cluster_dict:
                
                if cluster_intra_key != cluster_key:
                    
                    distance_list = []
                    for intra_cluster_counter in range(len(cluster_dict[cluster_intra_key])):
                        
                        graph_intra, node_intra = cluster_dict[cluster_intra_key][intra_cluster_counter]
                        vector_2 = list_graphs[graph_intra].nodes[node_intra]["sphere_3dcoords"]
                        
                        distance = np.linalg.norm(vector_1 - vector_2)
                        distance_list.append(distance)
                    
                    b_list.append(np.mean(distance_list))
                    
            b = np.min(b_list)
            
            silouhette = (b-a) / max(b,a)
            
            
            if cluster_key in result_dict:
                result_dict[cluster_key].append(silouhette)
            else:
                result_dict[cluster_key] = [silouhette]
                
    return result_dict


def get_texture_silhouette(silhouette_dict, centroid_dict, list_graphs, mesh):
    """
    Return the texture that gives the silhouette values for each cluster
    centroid
    """

    mesh_size = mesh.vertices.shape[0]

    texture = np.zeros(mesh_size)

    for cluster_key in centroid_dict:

        graph_num, node = centroid_dict[cluster_key]
        graph = list_graphs[graph_num]

        vertex = graph.nodes[node]["ico100_7_vertex_index"]

        silhouette = np.mean(silhouette_dict[cluster_key])

        texture[vertex] = silhouette + 1

    return texture

def get_silhouette_source_obj(silhouette_dict, centroid_dict, list_graphs, mesh):
    """
    Return a SourceObj that represent the silhouette value at each cluster centroid
    """

    nb_clusters = len(list(centroid_dict.keys()))
    silhouette_3Dpos = np.zeros((nb_clusters,3))
    silhouette_data = np.zeros(nb_clusters)

    # Get the data
    for cluster_i, cluster_key in enumerate(centroid_dict):

        graph_num, node = centroid_dict[cluster_key]
        graph = list_graphs[graph_num]

        vertex = graph.nodes[node]["ico100_7_vertex_index"]
        vertex_pos = mesh.vertices[vertex,:]
        #print(vertex_pos)
        #print(silhouette_3Dpos.shape)
        silhouette_3Dpos[cluster_i,:] = vertex_pos
        print(np.mean(silhouette_dict[cluster_key]) + 1)
        silhouette_data[cluster_i] = np.mean(silhouette_dict[cluster_key]) + 1

    # Create the source obj
    silhouette_text = [str(elem) for elem in silhouette_data]
    source_silhouette = SourceObj("source_silhouette",
                                  silhouette_3Dpos,
                                  silhouette_data,
                                  text=silhouette_text,
                                  radius_min = 20,
                                  radius_max = 30,
                                  text_size=50)

    return source_silhouette


def change_mean_sphere(sphere_mesh, mean):
    """
    Return the mesh of the sphere translated to a different mean than 0
    """
    
    radius = np.linalg.norm(sphere_mesh.vertices[1,:])
    vector_translation = mean - radius
    
    # add the vector needed to translate each vertex to the right point
    for i in range(sphere_mesh.vertices.shape[0]):
        sphere_mesh.vertices[i,:] += vector_translation
        
    return sphere_mesh

def get_sphere_silhouette(sphere_mean, silhouette_value, radius=2):
    """
    Return The mesh and texture of a sphere that represent the silhouette
    """

    # Generate the sphere
    sphere_mesh = sps.generate_sphere_icosahedron(subdivisions=3, radius=radius)

    # move the sphere to the right position
    vector_translation = sphere_mean #- radius
    for i in range(sphere_mesh.vertices.shape[0]):
        sphere_mesh.vertices[i,:] += vector_translation

    # generate the texture with the silhouette value
    sphere_texture = np.zeros(sphere_mesh.vertices.shape[0])
    sphere_texture += silhouette_value + 1

    # return the mesh and the texture
    return sphere_mesh, sphere_texture



def get_silhouette_texture_and_preview_it(path_to_graphs, path_mesh, path_silhouette, path_to_save, method):
    """
    Do the full pipeline : load the graphs and clusters,
    calculate the silhouette and centroid,
    calculate the texture and apply it on the mesh
    """

    # Get the graphs and their clusters
    list_graphs = load_graphs_in_list(path_to_graphs)
    cluster_dict = create_clusters_lists(list_graphs)

    # Get the mesh
    mesh_clustering = sio.load_mesh(path_mesh)
    transfo_full = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    mesh_clustering.apply_transform(transfo_full)

    # Calculate the centroid
    centroid_dict = get_centroid_clusters(list_graphs, cluster_dict)

    # Calculate or load the silhouette values
    if path_silhouette != "":
        pickle_in = open(path_silhouette,"rb")
        silhouette_dict = pickle.load(pickle_in)
        pickle_in.close()

    else:
        silhouette_dict = get_all_silhouette_value(list_graphs, cluster_dict)

    # save the silhouette value if necessary
    if path_to_save != "":
        pickle_out = open(path_to_save,"wb")
        pickle.dump(silhouette_dict, pickle_out)
        pickle_out.close()

    
    if method == "texture":
        # Calculate the texture with silhouette
        silhouette_texture = get_texture_silhouette(silhouette_dict, centroid_dict, list_graphs, mesh_clustering)

        # preview the texture
        visb_sc = visbrain_plot(mesh=mesh_clustering,
                                tex=silhouette_texture,
                                caption='Test du mesh',
                                cmap="jet")

    elif method == "source":
        silhouette_source = get_silhouette_source_obj(silhouette_dict,
                                                      centroid_dict,
                                                      list_graphs,
                                                      mesh_clustering)

        texture = np.zeros(mesh_clustering.vertices.shape[0])
        visb_sc = visbrain_plot(mesh = mesh_clustering,
                                tex = texture,
                                caption="text source obj",
                                cmap="jet")

        visb_sc.add_to_subplot(silhouette_source)

    elif method == "spheres":
        full_mesh = mesh_clustering
        full_texture = np.zeros(mesh_clustering.vertices.shape[0])

        for cluster_key in silhouette_dict:

            # get the right variables for the sphere generation
            graph_num, node = centroid_dict[cluster_key]
            vertex_centroid = list_graphs[graph_num].nodes[node]["ico100_7_vertex_index"]
            centroid_position = mesh_clustering.vertices[vertex_centroid,:]

            silhouette_value = np.mean(silhouette_dict[cluster_key])

            # get the mesh and the texture of the sphere
            sphere_mesh, sphere_texture = get_sphere_silhouette(centroid_position, silhouette_value, radius=3)

            # Add the sphere to the full mesh and do the same with texture
            full_mesh += sphere_mesh
            full_texture = np.concatenate((full_texture, sphere_texture))

        # load the values in the visbrain object
        visb_sc = visbrain_plot(mesh=full_mesh,
                                tex=full_texture,
                                caption='Test du mesh',
                                cmap="jet")

    visb_sc.preview()




if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Visualise the silhouette value of the clustering")
    parser.add_argument("path_to_folder",
                        help="path to a folder which contains the graphs")
    parser.add_argument("--path_mesh",
                        help="path to the mesh file",
                        default="/Users/Nathan/stageINT/stage_nathan/data_pits_graph/template/lh.OASIS_testGrp_average.gii")
    parser.add_argument("--path_silhouette",
                        help="Path to the file with precalculated silhouette values",
                        default="")
    parser.add_argument("--path_to_save",
                        help="Path to save the silhouette dict once it is calculated. If not prrovided, do not save the file",
                        default="")
    parser.add_argument("--method",
                        help="The method to use to represent the silhouette, either texture or source",
                        default="spheres")
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    path_mesh = args.path_mesh
    path_silhouette = args.path_silhouette
    path_to_save = args.path_to_save
    method = args.method

    get_silhouette_texture_and_preview_it(path_to_folder,
                                          path_mesh,
                                          path_silhouette,
                                          path_to_save,
                                          method)



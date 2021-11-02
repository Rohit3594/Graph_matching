''' This scripts aims at generating n simulated sulcal-pits graph
    Author : Nathan Buskulic
    Creation date : March 11th 2020  
'''

import argparse
import numpy as np
import networkx as nx
import slam.plot as splt
import slam.topology as stop
import slam.generate_parametric_surfaces as sgps
import trimesh
import os


def geodesic_distance_sphere(coord_a, coord_b, radius):
    ''' Return the geodesic distance of two 3D vectors on a sphere
    '''
    return radius * np.arccos(np.clip(np.dot(coord_a, coord_b) / np.power(radius,2),-1,1))


def graph_from_mesh(mesh):
    # Get the adjacency graph
    #########################################################
    # that fucking function is bugged!!!!!!!!!!!!!!!!!!!!
    # graph = sphere_random_sampling.vertex_adjacency_graph
    #########################################################
    adja = stop.edges_to_adjacency_matrix(mesh)
    graph = nx.from_numpy_matrix(adja.todense())
    # to be tested graph = nx.from_edgelist(sphere_random_sampling.edges_sorted)

    # Create dictionnary that will hold the attributes of each node
    node_attribute_dict = {}
    for node in graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(mesh.vertices[node])}

    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)

    # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0 # useful for affinity matrix caculation
    for edge in graph.edges:

        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id":id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)
    return graph


def generate_reference_graph(nb_vertices, radius):
    # Generate random sampling
    sphere_random = sgps.generate_sphere_random_sampling(vertex_number=nb_vertices, radius=radius)
    graph = graph_from_mesh(sphere_random)

    return graph


def generate_noisy_graph(original_graph, sigma_noise_nodes=1, nb_outliers=0, radius=100):

    # add the necessary nodes
    noisy_vertices = list()
    for node_info in list(original_graph.nodes(data=True)):
        noisy_coordinate = node_info[1]["coord"] + \
                           np.random.multivariate_normal(np.zeros(3), np.eye(3) * sigma_noise_nodes)
        # On projete ce point sur la sphère
        noisy_coordinate = noisy_coordinate / np.linalg.norm(noisy_coordinate) * radius
        noisy_vertices.append(noisy_coordinate)

    # noisy_sphere_no_o = sgps.tri_from_hull(noisy_vertices)
    # add outlier nodes
    if nb_outliers > 0:
        outlier_nodes = sphere_random_sampling(vertex_number=nb_outliers, radius=radius)
        noisy_vertices.extend(outlier_nodes)

    noisy_sphere = sgps.tri_from_hull(noisy_vertices)
    noisy_graph = graph_from_mesh(noisy_sphere)

    return noisy_graph


def sphere_random_sampling(vertex_number=10, radius=1.0):
    """
    random sampling of 3D coordinates of points on a sphere
    :param vertex_number: number of vertices in the output spherical mesh
    :param radius: radius of the output sphere
    :return:
    """
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    if radius != 1:
        coords = radius * coords
    return coords


def get_in_between_perm_matrix(perm_mat_1, perm_mat_2):
    """ 
    Given two permutation from noisy graphs to a reference graph,
    Return the permutation matrix to go from one graph to the other
    """

    result_perm = np.zeros((perm_mat_1.shape[0],), dtype=int)
    
    for node_reference, node_noisy_1 in enumerate(perm_mat_1):
        
        # get the corresponding node in the second graph
        node_noisy_2 = perm_mat_2[node_reference]
    
        # Fill the result
        result_perm[node_noisy_1] = node_noisy_2
        
    return result_perm

    
def generate_graph_family(reference_graph, nb_graphs, radius, nb_outliers, noise_node=1):
    """
    Generate n noisy graphs from a reference graph alongside the 
    ground truth permutation matrices.
    """

    nb_vertices = len(reference_graph)
    # Initialise permutation matrices to reference graph
    ground_truth_perm_to_ref = np.zeros((nb_graphs, nb_vertices), dtype=int)
    ground_truth_perm = np.zeros((nb_graphs, nb_graphs, nb_vertices), dtype=int)

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []
    ground_truth = range(len(reference_graph))
    # We generate the n noisy graphs
    for c_graph in range(nb_graphs):
        noisy_graph = generate_noisy_graph(reference_graph, noise_node, nb_outliers, radius=radius)
        
        # Save the ground truth permutation
        ground_truth_perm_to_ref[c_graph,:] = ground_truth

        # Save the graph
        list_noisy_graphs.append(noisy_graph)
        
    # We generate the ground_truth permutation  between graphs
    # since we do not permute nodes anymore, this is equivalent to
    # duplicate ground_truth_perm_to_ref
    for i_graph in range(nb_graphs):
        for j_graph in range(nb_graphs):
            ground_truth_perm[i_graph, j_graph, :] = \
                get_in_between_perm_matrix(ground_truth_perm_to_ref[i_graph,:], ground_truth_perm_to_ref[j_graph,:])

    return ground_truth_perm, list_noisy_graphs


def generate_n_graph_family_and_save(path_to_write, nb_runs, nb_graphs, nb_vertices, radius, list_noise, list_outliers, save_reference=0, reference_graph_in=None):
    ''' Generate n family of graphs for each couple (noise, outliers). The graphs are saved
        in a folder structure at the point path_to_write
    '''

    # check if the path given is a folder otherwise create one
    if not os.path.isdir(path_to_write):
        os.mkdir(path_to_write)

    # for each set of parameters
    for noise in list_noise:
        for outliers in list_outliers:

            folder_name = "nb_vertices_"+str(nb_vertices)+"_noise_"+str(noise)+"_outliers_"+str(outliers)
            path_parameters_folder = os.path.join(path_to_write,folder_name)

            # If the folder does not exist create one
            if not os.path.isdir(path_parameters_folder):
                os.mkdir(path_parameters_folder)


            # generate n families of graphs
            for i_graph in range(nb_runs):
                if reference_graph_in is None:
                    # Generate the reference graph
                    reference_graph = generate_reference_graph(nb_vertices=nb_vertices, radius=radius)
                else:
                    reference_graph = reference_graph_in
                ground_truth_perm, list_graphs = generate_graph_family(reference_graph=reference_graph,
                                                                       nb_graphs=nb_graphs,
                                                                       radius=radius, 
                                                                       nb_outliers=outliers, 
                                                                       noise_node=noise)

                # save the graph in a folder i_graph
                path_graph_folder = os.path.join(path_parameters_folder,str(i_graph))
                if not os.path.isdir(path_graph_folder):
                    os.mkdir(path_graph_folder)
                    os.mkdir(os.path.join(path_graph_folder,"graphs"))

                for i_family, graph_family in enumerate(list_graphs):
                    nx.write_gpickle(graph_family, os.path.join(path_graph_folder,"graphs","graph_"+str(i_family)+".gpickle"))
                if save_reference:
                    nx.write_gpickle(reference_graph, os.path.join(path_graph_folder, "graphs", "reference.gpickle"))
                
                np.save(os.path.join(path_graph_folder, "ground_truth"), ground_truth_perm)
                

if __name__ == '__main__':

    # # We parse the argument from command line
    # parser = argparse.ArgumentParser(description="Generate simulated sulcal pits graphs based on two variables : noise and outliers")
    # parser.add_argument("path_to_write", help="path where the folders will be generated")
    # parser.add_argument("--nb_runs", help="number of families of graph to generate per set of parameters", default=5, type=int)
    # parser.add_argument("--nb_graphs", help="number of graph to create for set of parameters", default=5, type=int)
    # parser.add_argument("--nb_vertices", help="number of vertices to create the graph with", default=90, type=int)
    # parser.add_argument("--min_noise", help="minimum noise value", default=1, type=float)
    # parser.add_argument("--max_noise", help="maximum noise value", default=2,  type=float)
    # parser.add_argument("--step_noise", help="step size for noise values", default=0.1, type=float)
    # parser.add_argument("--min_outliers", help="minimum outliers value", default=1, type=int)
    # parser.add_argument("--max_outliers", help="maximum outliers value", default=10, type=int)
    # parser.add_argument("--step_outliers", help="step size for outliers values", default=1, type=int)
    # parser.add_argument("--save_reference", help="Wether to save the reference graph or not. Should not be used unless for visualisation purpose. (0 or 1)", default=0, type=int)
    # args = parser.parse_args()
    #
    #
    # path_to_write = args.path_to_write
    # nb_runs = args.nb_runs
    # nb_graphs = args.nb_graphs
    # nb_vertices = args.nb_vertices
    # min_noise = args.min_noise
    # max_noise = args.max_noise
    # step_noise = args.step_noise
    # min_outliers = args.min_outliers
    # max_outliers = args.max_outliers
    # step_outliers = args.step_outliers
    # save_reference = args.save_reference


    #path_to_write = '/mnt/data/work/python_sandBox/stage_nathan/data/simu_graphs/with_ref_for_visu'
    path_to_write = '/home/rohit/PhD_Work/stage_nathan/data'
    #path_to_write = '/hpc/meca/users/auzias/ISBI2020_graph_matching/simu/generation_multi'
    nb_runs = 10
    nb_graphs = 3
    nb_vertices = 85
    min_noise = 50
    max_noise = 51
    step_noise = 20
    min_outliers = 10
    max_outliers = 11
    step_outliers = 10
    save_reference = 1
    # We define the parameters used throughout the script
    nb_vertices = nb_vertices
    radius = 100
    list_noise = np.arange(min_noise, max_noise, step_noise)
    list_outliers = np.array(list(range(min_outliers, max_outliers, step_outliers)))



    #reference_graph = nx.read_gpickle('/mnt/data/work/python_sandBox/stage_nathan/data/simu_graphs/with_ref_for_visu/noise_5.0,outliers_0_old/0/graphs/reference.gpickle')
    reference_graph = None
    # call the generation procedure
    generate_n_graph_family_and_save(path_to_write = path_to_write,
                                     nb_runs = nb_runs,
                                     nb_graphs = nb_graphs,
                                     nb_vertices = nb_vertices,
                                     radius = radius,
                                     list_noise = list_noise,
                                     list_outliers = list_outliers,
                                     save_reference = save_reference,
                                     reference_graph_in=reference_graph)


import networkx as nx
import numpy as np
import scipy.io as sio
import os
from multiprocessing import Pool, Lock
import subprocess
import argparse


def gaussian_kernel(attribute_1, attribute_2, gamma=1):
    ''' Return the value of a gaussian kernel between two attributes '''
    return np.exp(-gamma * np.power(np.linalg.norm(attribute_1 - attribute_2), 2))


def kernel(attribute_1, attribute_2, kernel_arg):
    """ Calculate the kernel of the two attributes given
        parameters in kernel_arg
    """

    kernel_type, kernel_dict = kernel_arg

    if kernel_type == "gaussian":
        if kernel_dict["attribute_type"] == "coord":
            return gaussian_kernel(attribute_1, attribute_2, gamma=kernel_dict["gaussian_gamma_coord"])
        elif kernel_dict["attribute_type"] == "geodesic":
            return gaussian_kernel(attribute_1, attribute_2, gamma=kernel_dict["gaussian_gamma_geodesic"])
    else:
        print("problème")
        return 0

    
def compute_median_distances_all_pair(list_vectors, distance_type="euclidean", radius=100):
    """ given a list of vectors, compute the median of the
        distances of all pair of values in the list """

    list_distances = []
    for i, elem_i in enumerate(list_vectors):
        for j in range(i+1, len(list_vectors)):
            elem_j = list_vectors[j]
            if distance_type == "euclidean":
                distance = np.linalg.norm(elem_i - elem_j)
            elif distance_type == "geodesic":
                distance = radius * np.arccos(np.clip(np.dot(elem_i,elem_j) / np.power(radius,2),-1, 1))
            else:
                print("distance type not allowed !", distance_type)

            list_distances.append(distance)

    return np.median(list_distances)
            
    


def compute_heuristic_gamma(graph_1, graph_2):
    """ Calculate a gamma value for the gaussian kernel
        based on a heuristic (take the median of all
        distances between pair of coordinates)
    """

    # calcule the heuristic for graph_1

    # we get all the coordinates for graph_1
    full_coordinates = []
    for node in graph_1.nodes:
        full_coordinates.append(graph_1.nodes[node]["coord"])
    graph_1_gamma_coord = compute_median_distances_all_pair(full_coordinates, "geodesic")

    # we get all the coordinates for graph_2
    full_coordinates = []
    for node in graph_2.nodes:
        full_coordinates.append(graph_2.nodes[node]["coord"])
    graph_2_gamma_coord = compute_median_distances_all_pair(full_coordinates, "geodesic")

    gamma_coord = 1/np.mean([graph_1_gamma_coord, graph_2_gamma_coord])


    # we get all the geodesic distances for graph_1
    full_coordinates = []
    for edge in graph_1.edges:
        full_coordinates.append(graph_1.edges[edge]["geodesic_distance"])
    graph_1_gamma_geo = compute_median_distances_all_pair(full_coordinates, "euclidean")

    # we get all the geodesic for graph_2
    full_coordinates = []
    for edge in graph_2.edges:
        full_coordinates.append(graph_2.edges[edge]["geodesic_distance"])
    graph_2_gamma_geo = compute_median_distances_all_pair(full_coordinates, "euclidean")

    gamma_geodesic = 1/np.mean([graph_1_gamma_geo, graph_2_gamma_geo])
        
    return gamma_coord, gamma_geodesic
    

    

def full_affinity(graph_1, graph_2, kernel_args):
    """ Calculation of the affinity value of two graphs of same size with the kernel function provided
    """
    print("full affinity matrix :",graph_1.number_of_nodes(), graph_2.number_of_nodes(), graph_1.number_of_edges(), graph_2.number_of_edges())
    
    # Initialise affinity matrix with zeros
    affinity_matrix = np.zeros((np.power(graph_1.number_of_nodes(), 2), np.power(graph_2.number_of_nodes(),2)))
    
    # we fill the affinity matrix with the kernel values.
    
    # we loop over all the possible permutations
    for node_a in graph_1.nodes:
        for node_i in graph_2.nodes:
            for node_b in graph_1.nodes:
                for node_j in graph_2.nodes:
                    
                    # We check if we need to take the attributes of nodes or edge
                    if node_a == node_b and node_i == node_j:
                        # We take the node attributes.
                        attribute_1 = graph_1.nodes[node_a]["coord"]
                        attribute_2 = graph_2.nodes[node_i]["coord"]
                        
                        # calculate the kernel value of these attributes
                        kernel_args[1]["attribute_type"] = "coord"
                        value_kernel = kernel(attribute_1, attribute_2, kernel_args)
                        
                        # add this in the right place in the affinity_matrix
                        affinity_matrix[ node_a * graph_2.number_of_nodes() + node_i, node_b * graph_2.number_of_nodes() + node_j] \
                                            = value_kernel
                        
                        
                    else:
                        # we check that the edges exist on both side and if so add the value to the affinity matrix
                        if (node_a, node_b) in graph_1.edges and (node_i, node_j) in graph_2.edges:
                            attribute_1 = graph_1.edges[(node_a, node_b)]["geodesic_distance"]
                            attribute_2 = graph_2.edges[(node_i, node_j)]["geodesic_distance"]

                            # get the kernel value
                            kernel_args[1]["attribute_type"] = "geodesic"
                            value_kernel = kernel(attribute_1, attribute_2, kernel_args)
                            affinity_matrix[ node_a * graph_2.number_of_nodes() + node_i, node_b * graph_2.number_of_nodes() + node_j] \
                                            = value_kernel
                            
    return affinity_matrix




def edge_to_edge_affinity(graph_1, graph_2, kernel_args):
    """calculate the edge_to_edge affinity matrix for kerGM"""
    
    # Initialise affinity matrix to zeros
    affinity_matrix = np.zeros((graph_1.number_of_edges(), graph_2.number_of_edges()))

    # we add the necessary attribute information
    kernel_args[1]["attribute_type"] = "geodesic"
    
    # We go through all combination of edges
    for edge_1 in graph_1.edges:
        for edge_2 in graph_2.edges:
            kernel_value = kernel(graph_1.edges[edge_1]["geodesic_distance"], graph_2.edges[edge_2]["geodesic_distance"], kernel_args)
            affinity_matrix[graph_1.edges[edge_1]["id"], graph_2.edges[edge_2]["id"]] = kernel_value
    
    return affinity_matrix




def node_to_node_affinity(graph_1, graph_2, kernel_args):
    """calculate the edge_to_edge affinity matrix for kerGM"""
    
    # Initialise affinity matrix to zeros
    affinity_matrix = np.zeros((graph_1.number_of_nodes(), graph_2.number_of_nodes()))

    # we add the necessary attribute arguments for the kernel
    kernel_args[1]["attribute_type"] = "coord"
    
    # We go through all combination of nodes
    for node_1 in graph_1.nodes:
        for node_2 in graph_2.nodes:
            kernel_value = kernel(graph_1.nodes[node_1]["coord"], graph_2.nodes[node_2]["coord"], kernel_args)
            affinity_matrix[node_1, node_2] = kernel_value
    
    return affinity_matrix


def get_head_and_tail_incidence_matrix(graph):
    """ From a given graph return a head and a tail incidence matrix where the direction
        of the edges have been chosen randomly
    """
    
    # initialize H (tail-incidence) and G(head-incidence) matrix. Notation are from KerGM paper
    H = np.zeros((graph.number_of_nodes(), graph.number_of_edges()))
    G = np.zeros((graph.number_of_nodes(), graph.number_of_edges()))
    
    for edge in graph.edges:
        
        # randomly select edge direction
        if np.random.rand(1) > 0.5:
            head = edge[0]
            tail = edge[1]
        else:
            head = edge[1]
            tail = edge[0]
            
        # add the information to the matrices
        edge_id = graph.edges[edge]["id"]
        H[head, edge_id] = 1
        G[tail, edge_id] = 1
        
    return H, G


def generate_affinity_and_incidence_for_pair(graph_1, graph_2, kernel_args):
    """ Generate the different affinity matrix as well as the incidences matrix
        necessary for the testing of different algorithms (like KerGM)
    """

    # Generate affinity matrices
    full_affinity_matrix = full_affinity(graph_1, graph_2, kernel_args)
    kE11 = edge_to_edge_affinity(graph_1, graph_1, kernel_args)
    kE22 = edge_to_edge_affinity(graph_2, graph_2, kernel_args)
    kE12 = edge_to_edge_affinity(graph_1, graph_2, kernel_args)
    kN12 = node_to_node_affinity(graph_1, graph_2, kernel_args)
    
    # generate head and tail incidence matrix
    H1, G1 = get_head_and_tail_incidence_matrix(graph_1)
    H2, G2 = get_head_and_tail_incidence_matrix(graph_2)

    return full_affinity_matrix, kE11, kE22, kE12, kN12, H1, G1, H2, G2


def load_generate_and_save_affinity_and_incidence_for_pair(path_to_folder, kernel_args):
    """ Generate the affinity and incidences matrix and save them
        in a given repositery
    """

    print(path_to_folder)
    
    # get the two graphs
    graph_1 = nx.read_gpickle(os.path.join(path_to_folder, "ref_graph.gpickle"))
    graph_2 = nx.read_gpickle(os.path.join(path_to_folder, "noisy_graph.gpickle"))

    # if the kernel is gaussian get the gamma value for the coordinate
    # and the geodesic distance
    if kernel_args[0] == "gaussian" and kernel_args[1]["gaussian_gamma"] == 0:
        gamma_coord, gamma_geodesic = compute_heuristic_gamma(graph_1, graph_2)
        print("gamma coord:",gamma_coord,"gamma geo:", gamma_geodesic)
        kernel_args[1]["gaussian_gamma_coord"] = gamma_coord
        kernel_args[1]["gaussian_gamma_geodesic"] = gamma_coord

    # generate all the necessary informations
    full_affinity_matrix, kE11, kE22, kE12, kN12, H1, G1, H2, G2 \
        = generate_affinity_and_incidence_for_pair(graph_1, graph_2, kernel_args)

    # Create dictionnaries that hold the information
    dict_affinity_to_save = {"full_affinity":full_affinity_matrix, "kE11":kE11, "kE22":kE22, "kE12": kE12, "kN12":kN12}
    dict_incidence_matrix_to_save = {"H1":H1,"H2":H2,"G1":G1,"G2":G2}

    # Save everything in an appropriate matlab format
    sio.savemat(os.path.join(path_to_folder, "affinity.mat"), dict_affinity_to_save, do_compression=True)
    sio.savemat(os.path.join(path_to_folder, "incidence_matrices.mat"), dict_incidence_matrix_to_save, do_compression=True)


    

def do_the_pairwise_calculation(path_to_run, path_to_graphs, graph_i, graph_j, kernel_args, lock):


    # Get the two graphs
    graph_1 = nx.read_gpickle(os.path.join(path_to_graphs, "graph_"+str(graph_i)+".gpickle"))
    graph_2 = nx.read_gpickle(os.path.join(path_to_graphs, "graph_"+str(graph_j)+".gpickle"))

    if kernel_args[0] == "gaussian" and kernel_args[1]["gaussian_gamma"] == 0:
        gamma_coord, gamma_geodesic = compute_heuristic_gamma(graph_1, graph_2)
        print("gamma coord:",gamma_coord,"gamma geo:", gamma_geodesic)
        kernel_args[1]["gaussian_gamma_coord"] = gamma_coord
        kernel_args[1]["gaussian_gamma_geodesic"] = gamma_geodesic

        
    full_affinity_matrix, kE11, kE22, kE12, kN12, H1, G1, H2, G2 \
        = generate_affinity_and_incidence_for_pair(graph_1, graph_2, kernel_args)

    
    dict_affinity_to_save = {"full_affinity":full_affinity_matrix, "kE11":kE11, "kE22":kE22, "kE12": kE12, "kN12":kN12}
    dict_incidence_matrix_to_save = {"H1":H1,"H2":H2,"G1":G1,"G2":G2}

    
    # save it in a temporary folder
    # create a folder if it doesn't exist
    path_to_save_folder = os.path.join(path_to_run, "pairwise_matrices", str(graph_i)+"_"+str(graph_j))
    if not os.path.isdir(path_to_save_folder):
        os.mkdir(path_to_save_folder)

    print(path_to_save_folder)
    sio.savemat(os.path.join(path_to_save_folder, "affinity.mat"), dict_affinity_to_save, do_compression=True)
    sio.savemat(os.path.join(path_to_save_folder, "incidence_matrices.mat"), dict_incidence_matrix_to_save, do_compression=True)

    # Run matlab to get the results in a subfolder
    path_to_root_project = "/hpc/meca/users/buskulic.n/stage_nathan"
    path_to_pairwise_algo = os.path.join(path_to_root_project, "pairwise_algorithms/matlab_implem/KerGM_code_Nathan")
    #path_to_pairwise_algo="${path_to_root_project}pairwise_algorithms/matlab_implem/KerGM_code_Nathan"

    lock.acquire()
    subprocess.run(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", "addpath(genpath('"+path_to_pairwise_algo+"'));path_to_folder='"+path_to_save_folder+"/';run('"+path_to_pairwise_algo+"/launch_and_save_pairwise_matching_algorithms_for_multi.m');exit;"])
    lock.release()

    print("addpath(genpath('",path_to_pairwise_algo,"'));path_to_folder='",path_to_save_folder,"';run('",path_to_pairwise_algo,"/launch_and_save_pairwise_matching_algorithms_for_multi.m');exit;")
    #$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_pairwise_algo}'));path='${1}';run('${path_to_pairwise_algo}/full_pairwise_for_multigraph.m');exit;"

    # remove the affinity ?
    os.remove(os.path.join(path_to_save_folder,"affinity.mat"))
    
    # reconstitute the full matrix ?
    
    
def launch_and_save_pairwise_algos_from_multi(path_to_read, nb_workers, kernel_args):

    list_jobs = []
    lock = Lock()

    for params_path in os.listdir(path_to_read):
        for run_path in os.listdir(os.path.join(path_to_read,params_path)):

            path_to_graphs = os.path.join(path_to_read, params_path, run_path, "graphs")

            nb_graphs = len([elem for elem in os.listdir(path_to_graphs) if elem[0] == "g"])

            path_to_run = os.path.join(path_to_read, params_path, run_path)

            path_to_pairwise_subfolder = os.path.join(path_to_run, "pairwise_matrices")
            if not os.path.isdir(path_to_pairwise_subfolder):
                os.mkdir(path_to_pairwise_subfolder)

            for graph_i in range(0, nb_graphs - 1):
                for graph_j in range(graph_i + 1, nb_graphs):

                    list_jobs.append((path_to_run, path_to_graphs, graph_i, graph_j, kernel_args, lock))
                    

            
    with Pool(processes=nb_workers) as pool:

        pool.starmap(do_the_pairwise_calculation, list_jobs)
            

kernel_type = "gaussian"
kernel_args = (kernel_type, {"gaussian_gamma":0})
path_to_read = "/Users/Nathan/stageINT/stage_nathan/generation_graphes/generation_multi/test_change_node_generation"
path_to_read="/hpc/meca/users/buskulic.n/stage_nathan/generation_graphes/generation_multi/test_for_pairwise"
launch_and_save_pairwise_algos_from_multi(path_to_read, 1, kernel_args)

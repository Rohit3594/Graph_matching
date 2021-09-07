import numpy as np
import pickle
import os
import scipy.io as sio
import networkx as nx


def get_bulk_matrix_for_one_algorithm(path_to_run, name_algo, transpose):

    path_graphs = os.path.join(path_to_run, "graphs")

    nb_graphs = len([elem for elem in os.listdir(path_graphs) if elem[0] == "g"])
    size_graph = nx.read_gpickle(os.path.join(path_graphs, "graph_0.gpickle")).number_of_nodes()

    path_pairwise_matrices = os.path.join(path_to_run, "pairwise_matrices")

    full_matrix = np.eye(nb_graphs * size_graph)

    for graph_pair in os.listdir(path_pairwise_matrices):

        graph_decomposition = graph_pair.split("_")
        graph_i = int(graph_decomposition[0])
        graph_j = int(graph_decomposition[1])

        path_graph_pair = os.path.join(path_pairwise_matrices, graph_pair)
        path_sub_matrix = os.path.join(path_graph_pair, "X_"+name_algo+".mat")

        
        # load the matrix
        if os.path.exists(path_sub_matrix):
            sub_matrix = sio.loadmat(path_sub_matrix, appendmat=False)["X"]
        else:
            print("OUPS")
            print(path_sub_matrix)
            sub_matrix = np.eye(size_graph)

        if transpose:
            sub_matrix = np.transpose(sub_matrix)

        # add this matrix in the full matrix
        a = size_graph * graph_i
        b = size_graph * (graph_i + 1)
        c = size_graph * graph_j
        d = size_graph * (graph_j + 1)

        full_matrix[a:b, c:d] = sub_matrix
        full_matrix[c:d,a:b] = np.transpose(sub_matrix)

    return full_matrix


def get_bulk_matrix_for_every_run(path_to_read, name_algo, transpose):

    print(path_to_read)
    for params in os.listdir(path_to_read):
        for run in os.listdir(os.path.join(path_to_read, params)):
            path_to_run = os.path.join(path_to_read, params, run)
            print(path_to_run)

            full_matrix = get_bulk_matrix_for_one_algorithm(path_to_run, name_algo, transpose)
            dict_to_save = {"full_assignment_mat" : full_matrix}
            
            path_to_save = os.path.join(path_to_run, "X_pairwise_"+name_algo)
            if name_algo == "kergm":
                path_to_save += "_2"
            path_to_save += ".mat"
            sio.savemat(path_to_save, dict_to_save, do_compression = True)


path_to_read = "/hpc/meca/users/buskulic.n/stage_nathan/generation_graphes/generation_multi/simus_not_complete_test"
get_bulk_matrix_for_every_run(path_to_read, "ipf", True)
get_bulk_matrix_for_every_run(path_to_read, "smac", True)
get_bulk_matrix_for_every_run(path_to_read, "rrwm", True)
get_bulk_matrix_for_every_run(path_to_read, "kergm", False)

        

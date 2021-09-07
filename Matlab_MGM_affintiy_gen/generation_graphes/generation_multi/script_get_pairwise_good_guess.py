"""
This scripts aims at generating pairwise matching between a family of graph
using good guess.
Author : Nathan Buskulic
Creation date : April 2nd 2020
"""

import numpy as np
import networkx as nx
import scipy.io as sio
from multiprocessing import Pool
import argparse
import os

def paiwise_nearest_points_matching(graph_1, graph_2, radius=100):
    """
    Return a matching between two graphs by matching points that
    are near one another
    """

    # We randomise the order in which we go through the nodes
    order_nodes = list(range(graph_1.number_of_nodes()))
    np.random.shuffle(order_nodes)
    
    # We initialise the list that tells us which nodes from
    # graph 2 is already taken (1 means available, 0 means taken)
    available_nodes = np.ones((graph_2.number_of_nodes()), dtype=int)
    
    # Initialize the matching matrix
    matching_mat = np.zeros((graph_1.number_of_nodes(),graph_2.number_of_nodes()))
    
    for node_1 in order_nodes:
        
        # We randomize the order in which we look at the nodes
        # in the second graph
        order_nodes_2 = list(range(graph_2.number_of_nodes()))
        np.random.shuffle(order_nodes_2)
        
        minDist = -1
        for node_2 in order_nodes_2:
            
            # if the node is still available
            if available_nodes[node_2] == 1:
                
                dist = np.linalg.norm(graph_1.nodes[node_1]["coord"] - graph_2.nodes[node_2]["coord"])
                
                if dist < minDist or minDist == -1:
                    chosen_node = node_2
                    minDist = dist
                    
        # We update the matching matrix
        matching_mat[node_1, chosen_node] = 1
        available_nodes[chosen_node] = 0
        
    return matching_mat


def generate_and_save_full_pairwise_good_guess(path_to_folder):
    """
    Given a path to a specific folder with graph family, return
    and save the full pairwise matrix of good guesses based on the
    distance between points.
    """

    # We get the graphs
    path_to_graphs = os.path.join(path_to_folder,"graphs")
    nb_graphs = len(os.listdir(path_to_graphs))
    print("number_of_graphs:",nb_graphs)

    # get the number of nodes
    print(os.path.join(path_to_graphs, "graph_0.gpickle"))
    graph_r = nx.read_gpickle(os.path.join(path_to_graphs, "graph_0.gpickle"))
    nb_nodes = graph_r.number_of_nodes()

    # Initialise the result matrix
    full_pairwise_mat = np.eye(nb_graphs*nb_nodes)
    
    for graph_nb_1 in range(nb_graphs):
        # load the first graph
        graph_1 = nx.read_gpickle(os.path.join(path_to_graphs, "graph_"+str(graph_nb_1)+".gpickle"))
        
        for graph_nb_2 in range(graph_nb_1 + 1,nb_graphs):

            # Load the second graph
            graph_2 = nx.read_gpickle(os.path.join(path_to_graphs, "graph_"+str(graph_nb_2)+".gpickle"))

            # calculate the permutation matrix by good guess
            perm_mat = paiwise_nearest_points_matching(graph_1, graph_2)

            # fill the full pairwise matrix
            a = graph_nb_1 * nb_nodes
            b = a + nb_nodes
            c = graph_nb_2 * nb_nodes
            d = c + nb_nodes
            full_pairwise_mat[a:b,c:d] = perm_mat
            full_pairwise_mat[c:d,a:b] = np.transpose(perm_mat)

    
    # Save the matrix in the folder
    dict_to_save = {"full_assignment_mat":full_pairwise_mat}
    sio.savemat(os.path.join(path_to_folder,"X_pairwise_goodguess.mat"), dict_to_save)


def generate_and_save_all_pairwise_matrices(path_to_folder, nb_workers):
    """
    Given a root path to an experiment, go through every folder
    and generate all the good guess pairwise matrices
    """

    # Initialise the list of arguments
    list_arguments = []
    
    # for every set of parameters
    for parameters_name in os.listdir(path_to_folder):

        # For every run with these parameters
        for run_name in os.listdir(os.path.join(path_to_folder,parameters_name)):

            # Generate the pairwise matrixfor this folder
            path_to_run = os.path.join(path_to_folder,parameters_name,run_name)
            list_arguments.append(path_to_run)


    # Launch the processes
    with Pool(processes=nb_workers) as pool:
        pool.map(generate_and_save_full_pairwise_good_guess, list_arguments)
            
    

####################

if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Generate the pairwise good guess matrices for every folder in the experiment")
    parser.add_argument("path_to_folder", help="path to the experiment folder")
    parser.add_argument("--nb_workers", help="the number of threads to use to do the calculations", default=1, type=int)
    args = parser.parse_args()

    
    #path_to_folder = args.path_to_folder
    path_to_folder = '/hpc/meca/users/rohit/stage_nathan/generation_graphes/generation_multi/test_for_pairwise/'
    nb_workers = args.nb_workers
    
    generate_and_save_all_pairwise_matrices(path_to_folder, nb_workers)
    

    

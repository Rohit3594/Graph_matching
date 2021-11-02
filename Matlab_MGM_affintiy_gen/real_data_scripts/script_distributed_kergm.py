import numpy as np
import pickle
import scipy.io as sio
import subprocess as sp
from multiprocessing import Pool
import argparse
import os
import networkx as nx

def launch_kergm_calculation(path_to_folder, graph_number_1, graph_number_2, path_to_pairwise_kergm, path_to_matlab):
    """
    Launch a subprocess to calculate the matching between two graphs
    using KerGM
    """

    matlab_to_run = ("addpath(genpath('"+path_to_pairwise_kergm+"'));"
                     "launch_and_save_distributed_KerGM_matching('"+path_to_folder+"',"
                     "'"+str(graph_number_1)+"',"
                     "'"+str(graph_number_2)+"',"
                     "'affinity',"
                     "'incidence');"
                     "exit;")

    argument_function = [path_to_matlab, "-nodisplay", "-nosplash", "-nodesktop", "-r", matlab_to_run]

    # launch the subprocess
    sp.run(argument_function)


def launch_distributed_kergm_calculation(path_to_folder, path_to_pairwise_kergm, path_to_matlab, nb_workers):
    """
    Launch nb_workers kergm calculation at the same time
    """

    # Check that the folder name doen't end with /
    if path_to_folder[-1] == "/":
        path_to_folder = path_to_folder[:-1]

    # create the directory to save the kergm result if it does not exist
    path_to_kergm_saves = os.path.join(path_to_folder, "KerGM_results")
    if not os.path.exists(path_to_kergm_saves):
        os.makedirs(path_to_kergm_saves)

    # Get the number of graphs
    path_graphs = os.path.join(path_to_folder, "modified_graphs")
    nb_graphs = len(os.listdir(path_graphs))

    # Create the list with all the function to launch
    list_processes_to_launch = []
    
    for graph_1 in range(nb_graphs - 1):
        for graph_2 in range(graph_1+1, nb_graphs):

            argument_to_launch = (path_to_folder,
                                  graph_1,
                                  graph_2,
                                  path_to_pairwise_kergm,
                                  path_to_matlab)
            list_processes_to_launch.append(argument_to_launch)

    # launch the workers
    with Pool(nb_workers) as p:
        p.starmap(launch_kergm_calculation, list_processes_to_launch)


def get_bulk_matching_matrix(path_to_folder, path_to_pairwise_kergm, path_to_matlab, nb_workers):
    """
    Launch the calculation of the pairwise matching using kergm
    And then load them into a bulk matrix that is saved in the original folder
    """

    # Launch the calculation of pairwise matching
    launch_distributed_kergm_calculation(path_to_folder,
                                         path_to_pairwise_kergm,
                                         path_to_matlab,
                                         nb_workers)

    # Get the number of graphs
    path_graphs = os.path.join(path_to_folder, "modified_graphs")
    nb_graphs = len(os.listdir(path_graphs))

    # get the number of nodes of one graph (by looking at one graph)
    graph_0 = nx.read_gpickle(os.path.join(path_graphs,"graph_0.gpickle"))
    nb_nodes = graph_0.number_of_nodes()
    
    
    # initialise the bulk matrix
    tot_number_nodes = nb_graphs * nb_nodes
    bulk_matrix = np.zeros((tot_number_nodes, tot_number_nodes))

    # Read the kergm pairwise matching and fill the bulk matrix
    path_pairwise_matching = os.path.join(path_to_folder, "KerGM_results")
    for graph_1 in range(nb_graphs - 1):
        for graph_2 in range(graph_1 + 1, nb_graphs):

            # load the matching
            matching_name = "kergm_"+str(graph_1)+"_"+str(graph_2)+".mat"
            path_to_matching = os.path.join(path_pairwise_matching, matching_name)

            pairwise_matching = sio.loadmat(path_to_matching, appendmat=False)["X"]

            # Fill the bulk matrix
            a = nb_nodes * graph_1
            b = nb_nodes * graph_2
            bulk_matrix[a:a+nb_nodes, b:b+nb_nodes]

    # Add the transpose of the results to get a full matrix
    bulk_matrix = bulk_matrix + np.transpose(bulk_matrix) + np.eye(tot_number_nodes)

    # save the bulk matrix
    dict_to_save = {"full_assignment_mat": bulk_matrix}
    path_to_save = os.path.join(path_to_folder,"X_pairwise_kergm.mat")
    sio.savemat(path_to_save, dict_to_save, appendmat=False, do_compression=True)



    


##### Check some stuff out """""

if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Calculate in a distributed fashion Kergm matching")
    parser.add_argument("path_to_folder", help="path where the folders contains the graphs")
    parser.add_argument("--path_kergm",
                        help="path to the kergm algorithm",
                        default="/Users/Nathan/stageINT/stage_nathan/pairwise_algorithms/matlab_implem/KerGM_code_Nathan")
    parser.add_argument("--path_matlab",
                        help="path to matlab executable",
                        default="/Applications/MATLAB_R2018a.app/bin/matlab")
    parser.add_argument("--nb_workers",
                        help="number of threads to use for the distributed calculation",
                        default=4,
                        type=int)
    args = parser.parse_args()

    
    path_to_folder = args.path_to_folder
    path_kergm = args.path_kergm
    path_matlab = args.path_matlab
    nb_workers = args.nb_workers

    # start the processus
    get_bulk_matching_matrix(path_to_folder,
                             path_kergm,
                             path_matlab,
                             nb_workers)
    

import argparse
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
import scipy.io as sio
import pickle
import os


def get_consistency_cost_matrix(matching_matrix, nb_graphs, nb_nodes):
    """
    Based on a multi-matching matrix, create nb_graphs labelling each one based on a different 
    reference graph. Then compute for each nodes with which one it is associated and return the 
    associated statistic
    """
    

            
    cost_mat = np.zeros((nb_graphs * nb_nodes, nb_graphs * nb_nodes), dtype=int)
    
    # Create each universe with a reference graph
    for graph_ref_num in range(nb_graphs):
        
        # Initialise the universe
        universe = [[] for i in range(nb_nodes)]
        
        # For each other available graph create the labelling
        for other_graph in range(nb_graphs):
            
            # We take care of node calculating the result for the reference graph and itself
            # Get the permutation matrix from the reference graph to the other graph
            permutation_mat = matching_matrix[graph_ref_num * nb_nodes:(graph_ref_num+1)*nb_nodes, other_graph * nb_nodes:(other_graph+1)*nb_nodes]
                
            # Add the statistic to the result dict
            x_nz, y_nz = np.nonzero(permutation_mat)
            
            for i_x in range(len(x_nz)):
                x,y = x_nz[i_x], y_nz[i_x]
                
                # Fill the universe
                universe[x].append((other_graph,y))
               
        # Fill the dict result with the universe information
        for x_uni in range(len(universe)):
            
            sub_universe = universe[x_uni]
            
            for i_f_t in range(len(sub_universe)):
                first_tuple = sub_universe[i_f_t]
                first_node = first_tuple[0] * nb_nodes + first_tuple[1]
                
                for i_s_t in range(i_f_t + 1, len(sub_universe)):
                    second_tuple = sub_universe[i_s_t]
                    second_node = second_tuple[0] * nb_nodes + second_tuple[1]
                    
                    cost_mat[first_node, second_node] -= 1
                    cost_mat[second_node, first_node] -= 1
                    
    cost_mat = cost_mat - np.min(cost_mat)
                
    return cost_mat - np.eye(cost_mat.shape[0]) * np.max(cost_mat)


def get_matching_from_labelling(nb_nodes, nb_graphs, clustering_labels):
    """
    Transform the labelling into a bulk matrix
    that represent the matching.
    """

    nb_tot_node = nb_graphs * nb_nodes

    # initialise bulk_matrix
    bulk_permutation_matrix = np.zeros((nb_tot_node,nb_tot_node))

    # fill the matrix
    for i in range(clustering_labels.shape[0]):
        for j in range(i+1, clustering_labels.shape[0]):

            if clustering_labels[i] == clustering_labels[j] and clustering_labels[i] != -1:
                graph_i = i // nb_nodes
                graph_j = j // nb_nodes
                node_i = i % nb_nodes
                node_j = j % nb_nodes

                if graph_i != graph_j:

                    bulk_permutation_matrix[graph_i * nb_nodes + node_i, graph_j * nb_nodes + node_j] = 1

                    
    bulk_permutation_matrix = bulk_permutation_matrix + bulk_permutation_matrix.T + np.eye(bulk_permutation_matrix.shape[0])
    return bulk_permutation_matrix




def get_labeling_from_matching(matching, nb_graphs, eps=0.5, min_sample=0.2):
    """
    Return a labeling from a matching bulk matrix using DBSCAN 
    -------
    eps: float [0,1]
        Tells DBSCAN what perecentage of the max distance is used in the algorithm
    min_sample: float[0,1]
        Tell DBSCAN what percentage of the number of graphs will be used as the
        min_sample parameters in DBSCAN
    """

    eps_param = eps * np.max(matching)
    print("eps_param", eps_param)
    min_sample_param = int(min_sample * nb_graphs)
    print("min_s", min_sample_param)

    clustering = DBSCAN(eps=eps_param, min_samples=min_sample_param, metric="precomputed").fit(matching)
    return clustering.labels_



def get_and_write_labelling_matching(path_to_read, param_eps=0.5, param_min_sample=0.2):
    """
    Load all the necessary information and save the dict result in the right place
    """

    # load the mALS results
    matching_mALS = sio.loadmat(os.path.join(path_to_read,"X_mSync.mat"))["X"]

    # Get the number of graphs by looking into the modified_graph folder
    nb_graphs = len(os.listdir(os.path.join(path_to_read, "modified_graphs_msync")))

    # get the associated number of nodes
    nb_nodes = int(matching_mALS.shape[0]/nb_graphs)

    # calculate the cost matrix
    cost_mat = get_consistency_cost_matrix(matching_mALS, nb_graphs, nb_nodes)
    # save the cost matrix
    file_name_cost_mat = "cost_matrix_clustering_msync.npy"
    np.save(os.path.join(path_to_read,file_name_cost_mat), cost_mat)
    print("Graph view saved !")

    # calculate the labelling and the associated matching
    labelling = get_labeling_from_matching(cost_mat, nb_graphs, eps=param_eps, min_sample=param_min_sample)
    matching = get_matching_from_labelling(nb_nodes, nb_graphs, labelling)

    # Save the matching and the labelling
    file_name_matching = "X_clustering_matching_msync.npy"
    np.save(os.path.join(path_to_read,file_name_matching), matching)
    file_name_labelling = "X_clustering_labelling_msync.npy"
    np.save(os.path.join(path_to_read,file_name_labelling), labelling)



if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Generate the result tensor (accuracy) of the labels generated through a clustering mean (DBSCAN)")
    parser.add_argument("path_to_read", help="path where the folders contains the graphs and the algorithms results")
    parser.add_argument("--param_eps", help="value between 0 and 1 for the eps param of DBSCAN", default=0.5, type=float)
    parser.add_argument("--param_min_sample", help="value between 0 and 1 for the min_sample param of DBSCAN", default=0.3, type=float)
    
    args = parser.parse_args()

    
    path_to_read = args.path_to_read
    param_eps = args.param_eps
    param_min_sample = args.param_min_sample

    get_and_write_labelling_matching(path_to_read,
                                     param_eps,
                                     param_min_sample)

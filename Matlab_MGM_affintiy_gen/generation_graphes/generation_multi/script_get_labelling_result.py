import numpy as np
import os
import argparse
import pickle

def get_specs_params_with_dbscan(path_to_folder):
    """ go through all the folders to determine the possible
        parameters and assign them an integer value
    """
    
    result = {}
    
    for folder_name in os.listdir(path_to_folder):
        list_splitted = folder_name.split(",")
        
        for param_num, elem_param in enumerate(list_splitted):
            param_name, param_value = elem_param.split("_")
            param_value = float(param_value)
            
            # Create the parameter entries in the result
            if param_name not in result:
                result[param_name] = []
                
            if param_value not in result[param_name]: 
                result[param_name].append(param_value)
                
        # path to the clustering directory
        path_to_clustering = os.path.join(path_to_folder, folder_name, "0", "clustering_matching")
        for clustering_file in os.listdir(path_to_clustering):
            splitted_file_name = clustering_file[len("X_clustering_"):-4]
            for param_num, elem_param in enumerate(splitted_file_name.split(",")):
                param_name, param_value = elem_param.split("_")
                param_value = float(param_value)
                
                # Create the parameter entries in the result
                if param_name not in result:
                    result[param_name] = []
                
                if param_value not in result[param_name]: 
                    result[param_name].append(param_value)
                    
    # We sort the dict for later plotting purposes
    for param_name in result:
        result[param_name].sort()
        
    return result




def get_accuracy_permutation(target_permutation, given_permutation, use_precision=False):
    """ Return the accuracy of a given permutation knowing the ground truth permutation
        - ground_truth_permutation is a np array where the first axis corresponds to one graph of the family
        the second axis corresponds to another graph and the third axis contains the
        ground truth correspondence matrix between the first and second graph
        - given_permutation is a bulk matrix where all the permutation marix of the algorithm
        are put together.
    """
    
    count_ok = 0
    count_total = 0

    # Get automatically the number of graphs if not given in the first place
    nb_graphs = target_permutation.shape[0]
    nb_nodes = int(given_permutation.shape[0] / nb_graphs)

    for graph_1 in range(nb_graphs):
        for graph_2 in range(nb_graphs):
            
            # If we are looking at two different graphs we compute the accuracy
            if graph_1 != graph_2:
                
                ground_truth_list = target_permutation[graph_1, graph_2, :]
                sub_permutation = given_permutation[graph_1 * nb_nodes: (graph_1 + 1) * nb_nodes, graph_2 * nb_nodes: (graph_2 + 1) * nb_nodes]

                for node_1, node_2 in enumerate(ground_truth_list):
                    if sub_permutation[node_1, node_2] == 1:
                        count_ok += 1

                # Choose the metric to use
                if not use_precision:
                    count_total += len(ground_truth_list)
                else:
                    count_total += np.nonzero(sub_permutation)[0].shape[0]

    return count_ok/count_total




def get_and_write_results(path_to_read, path_to_write, use_precision=False):
    """
    Load all the necessary information and save the dict result in the right place
    """

    # Get the dict of parameters
    dict_parameters_correspondence = get_specs_params_with_dbscan(path_to_read)

    # We need to find how many runs were used so we go though the first folder
    nb_runs = len(os.listdir(os.path.join(path_to_read, os.listdir(path_to_read)[0])))

    # initialise result tensor
    result_tensor_shape = [len(dict_parameters_correspondence[param_name]) for param_name in dict_parameters_correspondence]\
                          + [nb_runs]
    result_tensor = np.zeros(result_tensor_shape)


    # We go through all the folders with the assumption that we know the parameters
    for i_noise, noise_val in enumerate(dict_parameters_correspondence["noise"]):
        for i_outliers, outliers_val in enumerate(dict_parameters_correspondence["outliers"]):
            for i_run in range(nb_runs):

                 # Get the ground truth
                path_to_ground_truth  = os.path.join(path_to_read,
                                                     "noise_"+str(noise_val)+",outliers_"+str(int(outliers_val)),
                                                     str(i_run),
                                                     "ground_truth.npy"
                )

                ground_truth = np.load(path_to_ground_truth)

                # Get the clustering result
                for i_eps, eps_val in enumerate(dict_parameters_correspondence["eps"]):
                    for i_minsamp, minsamp_val in enumerate(dict_parameters_correspondence["minsamp"]):

                        # Create the path to the clustering file
                        path_to_clustering_file = os.path.join(path_to_read,
                                                               "noise_"+str(noise_val)+",outliers_"+str(int(outliers_val)),
                                                               str(i_run),
                                                               "clustering_matching",
                                                               "X_clustering_eps_"+str(eps_val)+",minsamp_"+str(minsamp_val)+".npy"
                        )

                        # Load the file
                        matching = np.load(path_to_clustering_file)

                        accuracy = get_accuracy_permutation(ground_truth, matching, use_precision=use_precision)

                        # Fill the result tensor
                        result_tensor[i_noise, i_outliers, i_eps, i_minsamp, i_run] = accuracy


    # Write the result
    # save the result tensor
    dict_to_be_saved = {"parameters_correspondence":dict_parameters_correspondence, "DBSCAN":result_tensor}

    pickle_out = open(path_to_write,"wb")
    pickle.dump(dict_to_be_saved, pickle_out)
    pickle_out.close()




if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Generate the result tensor (accuracy) of the labels generated through a clustering mean (DBSCAN)")
    parser.add_argument("path_to_read", help="path where the folders contains the graphs and the algorithms results")
    parser.add_argument("path_to_write", help="path where to write the results")
    parser.add_argument("--use_precision", help="Whether or not the loss measure used is the recall or the precision. Only important for partial matching algorithms", default=0, type=int)
    args = parser.parse_args()

    
    path_to_read = args.path_to_read
    path_to_write = args.path_to_write
    use_precision = args.use_precision

    get_and_write_results(path_to_read,
                          path_to_write,
                          use_precision=use_precision
    )

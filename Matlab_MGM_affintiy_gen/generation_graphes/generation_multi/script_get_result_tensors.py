''' This scripts aims at generating n simulated sulcal-pits graph
    Author : Nathan Buskulic
    Creation date : March 11th 2020  
'''

import numpy as np
import scipy.io as sio
import pickle
import os
import argparse
import time



def get_noise_and_outliers_params(path_to_folder, number_parameters=2):
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
            
    # We sort the dict for later plotting purposes
    for param_name in result:
        result[param_name].sort()
        
    return result




def get_accuracy_permutation(target_permutation, given_permutation, sub_nb_graphs=None, use_precision=False, use_F1=False):
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
    if sub_nb_graphs == None:
        nb_graphs = target_permutation.shape[0]
    else:
        nb_graphs = sub_nb_graphs

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

                # If we want the recall measure
                if not use_precision:
                    count_total += len(ground_truth_list)
                elif use_precision:
                    count_total += np.nonzero(sub_permutation)[0].shape[0]
                    if count_total == 0:
                        return 0
                elif use_F1:
                    # get precision
                    count_total += np.nonzero(sub_permutation)[0].shape[0]
                    if count_total == 0:
                        precision = 0
                    else:
                        precision = np.clip(count_ok/count_total, 0, 1)

                    # get recall
                    count_total =  len(ground_truth_list)
                    recall = np.clip(count_ok/count_total, 0, 1)

                    return 2 * precision * recall / (precision + recall)

    return np.clip(count_ok/count_total, 0, 1)


def get_distance_between_matrices(matrix_1, matrix_2):
    """ 
    Return the number of differences between matrix_1 and matrix_2
    It is assumed that both matrices have the shape
    The values of each matrices are transformed in int format.
    """

    count_differences = 0

    x_ones, y_ones = np.nonzero(matrix_1)
    for i in range(len(x_ones)):
        if matrix_2[x_ones[i], y_ones[i]] == 0:
            count_differences += 1
    count_total = x_ones.shape[0]

    return count_differences / count_total



def get_result_tensor_for_given_algorithm(path_to_folder, param_correspondence_dict, name_of_result_file="X_kergm", transposed=False, sub_nb_graphs=None, use_precision=False, use_F1=False):
    """Go through all folders and build a 3D tensor that holds
    the accuracy metric for all set of parameters for a given algorithm.
    
    Arguments:
        path_to_folder: The path where the results have been calculated
        param_correspondence_dict: The result of the get_noise_and_outliers_params
    function that give an integer correspondence to each parameter value
        name_of_result_file: The file name to load in each folder (each one
    corresponds to a given algorithm)
    """

    # We split the path of the name of result file in case of given a path
    sub_folder_name, name_of_sub_file = os.path.split(name_of_result_file)
    
    # We need to find how many runs were used so we go though the first folder
    nbRuns = len(os.listdir(os.path.join(path_to_folder, os.listdir(path_to_folder)[0])))
    
    # We initialise the final tensor that hold the results

    result_tensor_shape = [len(param_correspondence_dict[param_name]) for param_name in param_correspondence_dict]\
                          + [nbRuns]
    result_tensor = np.zeros(result_tensor_shape)

    time_tensor = np.zeros(result_tensor_shape)
    
    #We go through all folders
    for parameter_folder in os.listdir(path_to_folder):
        
        # define the new path
        path_parameter_folder = os.path.join(path_to_folder, parameter_folder)
        
        # get the parameters
        splitted_param = parameter_folder.split(",")
        param_1_name, param_1_value = splitted_param[0].split("_")
        param_1_value = float(param_1_value)
        param_1_integer = param_correspondence_dict[param_1_name].index(param_1_value)
        
        param_2_name, param_2_value = splitted_param[1].split("_")
        param_2_value = float(param_2_value)
        param_2_integer = param_correspondence_dict[param_2_name].index(param_2_value)
        
        # We go through all the runs
        for run_i, run_folder in enumerate(os.listdir(path_parameter_folder)):
            
            path_run_folder = os.path.join(path_parameter_folder, run_folder)
            print(path_run_folder)
            
            # load the ground truth corespondence
            ground_truth = np.load(os.path.join(path_run_folder,"ground_truth.npy"))
            
            # load the algorithm result
            if name_of_result_file.find("X_pairwise") != -1:
                algorithm_res = sio.loadmat(os.path.join(path_run_folder,name_of_result_file+".mat"))["full_assignment_mat"]
            else:
                algorithm_res = sio.loadmat(os.path.join(path_run_folder,name_of_result_file+".mat"))["X"]
            
            # transpose it if necessary
            if transposed:
                algorithm_res = np.transpose(algorithm_res)

            # get the accuracy result
            accuracy = get_accuracy_permutation(ground_truth, algorithm_res, sub_nb_graphs=sub_nb_graphs, use_precision=use_precision, use_F1=use_F1)
            
            # add the result to the tensor
            result_tensor[param_1_integer, param_2_integer, run_i] = accuracy

            # Add the time to the tensor
            if name_of_result_file.find("X_pairwise") == -1:
                if sub_folder_name == "":
                    time = sio.loadmat(os.path.join(path_run_folder,"time"+name_of_sub_file[1:]+".mat"))["t"]
                    time_tensor[param_1_integer, param_2_integer, run_i] = time
                else:
                    time = sio.loadmat(os.path.join(path_run_folder,sub_folder_name,"time"+name_of_sub_file[1:]+".mat"))["t"]
                    time_tensor[param_1_integer, param_2_integer, run_i] = time
                    
    return (result_tensor, time_tensor)



def get_distance_mix_original(path_to_folder, param_correspondence_dict, name_result_file="X_mSync", mix_rate = 10):
    """Go through all folders and build a 3D tensor that holds
    the distance metric between the mixed version and the original one (calculated with KerGM)
    for all set of parameters for a given algorithm.
    
    Arguments:
        path_to_folder: The path where the results have been calculated
        param_correspondence_dict: The result of the get_noise_and_outliers_params
    function that give an integer correspondence to each parameter value
        name_result_file: The file name to load in each folder (each one
    corresponds to a given algorithm)
        mix_rate: the mix_rate that should be taken in account
    """
    
    # We need to find how many runs were used so we go though the first folder
    nbRuns = len(os.listdir(os.path.join(path_to_folder, os.listdir(path_to_folder)[0])))
    
    # We initialise the final tensor that hold the results
    result_tensor_shape = [len(param_correspondence_dict[param_name]) for param_name in param_correspondence_dict]\
                          + [nbRuns]
    result_tensor = np.zeros(result_tensor_shape)

    time_tensor = np.zeros(result_tensor_shape)
    
    #We go through all folders
    for parameter_folder in os.listdir(path_to_folder):
        
        # define the new path
        path_parameter_folder = os.path.join(path_to_folder, parameter_folder)
        
        # get the parameters
        splitted_param = parameter_folder.split(",")
        param_1_name, param_1_value = splitted_param[0].split("_")
        param_1_value = float(param_1_value)
        param_1_integer = param_correspondence_dict[param_1_name].index(param_1_value)
        
        param_2_name, param_2_value = splitted_param[1].split("_")
        param_2_value = float(param_2_value)
        param_2_integer = param_correspondence_dict[param_2_name].index(param_2_value)
        
        # We go through all the runs
        for run_i, run_folder in enumerate(os.listdir(path_parameter_folder)):
            
            path_run_folder = os.path.join(path_parameter_folder, run_folder)
            
            # load the ground truth corespondence
            ground_truth = np.load(os.path.join(path_run_folder,"ground_truth.npy"))

            # load the original result
            original_res = sio.loadmat(os.path.join(path_run_folder, name_result_file+".mat"))["X"]

            # load the mixed result
            mixed_result = sio.loadmat(os.path.join(path_run_folder, "results_mix", name_result_file+"_mix_"+str(mix_rate)+".mat"))["X"]

            # Get the distance between the two matrices
            distance = get_distance_between_matrices(original_res, mixed_result)
            
            # add the result to the tensor
            result_tensor[param_1_integer, param_2_integer, run_i] = distance
            
    return (result_tensor, time_tensor)



def get_and_write_all_results(path_to_read, path_to_write, method, use_subgraphs=False, use_precision=False, use_F1=False):
    """ Compute the result tensor for each algorithm and write
        the results in a given directory 
    """

    #load the dictionary of parameters integer correspondence
    dict_parameters_correspondence = get_noise_and_outliers_params(path_to_read)

    if method in ["KerGM","good_guess"]:
        # get the suffix for the method we want to save
        if method == "KerGM":
            suffix = ""
        elif method == "good_guess":
            suffix = "_good_guess"

        # If we are looking at the situation using every graph available
        if not use_subgraphs:
            # compute the result of all the given algorithms
            # Create the final dict to be saved
            dict_to_be_saved = {"parameter_correspondence" : dict_parameters_correspondence,
                                #"KerGM": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_kergm", False, ),
                                #"mSync": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_mSync"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                "mALS": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_mALS"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                #"mOpt": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_mOpt"+suffix, False),
                                #"cao":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao"+suffix, False),
                                #"cao_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_o"+suffix, False),
                                #"cao_s_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_s_o"+suffix, False),
                                #"cao_uc":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_uc"+suffix, False),
                                #"cao_uc_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_uc_o"+suffix, False),
                                #"cao_uc_s_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_uc_s_o"+suffix, False),
                                #"cao_pc":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_pc"+suffix, False),
                                #"cao_pc_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_pc_o"+suffix, False),
                                #"cao_pc_s_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_pc_s_o"+suffix, False),
                                #"cao_c":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_c"+suffix, False),
                                #"cao_c_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_c_o"+suffix, False),
                                #"cao_c_s_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_c_s_o"+suffix, False),
                                #"cao_cst":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_cst"+suffix, False),
                                #"cao_cst_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_cst_o"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                #"cao_cst_s_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_cao_cst_s_o"+suffix, False),
                                "ipf": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_ipf"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                "smac": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_smac"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                "rrwm": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_rrwm"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                #"KerGM_2": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_kergm_2"+suffix, False, use_precision=use_precision, use_F1=use_F1),
                                
            }

            # add the initial method
            if method == "KerGM":
                dict_to_be_saved["KerGM"] = get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_kergm", False, use_precision=use_precision, use_F1=use_F1)
            elif method == "good_guess":
                dict_to_be_saved["good_guess"] = get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_pairwise_goodguess", False, use_precision=use_precision)
            else:
                print("not an authorized pairwise method, something is wrong")

                
        # If we want to get the results with different numbers of graph used
        elif use_subgraphs:
            # initialize dict to be save
            dict_to_be_saved = {"parameter_correspondence" : dict_parameters_correspondence}

            # get the path to one specific experiment folder
            path_to_subgraphs = os.path.join(path_to_read, os.listdir(path_to_read)[0])
            path_to_subgraphs = os.path.join(path_to_subgraphs, "0")
            # We get the different numbers of graphs used
            nb_graphs_used = []
            for file_name in os.listdir(os.path.join(path_to_subgraphs, "results_sub_graphs")):

                current_graph_used = file_name.split("_")[-1]
                current_graph_used = int(current_graph_used[:current_graph_used.find(".")])

                if current_graph_used not in nb_graphs_used:
                    nb_graphs_used.append(current_graph_used)

            # We want to compy what has been done before for the mix method but in simpler term (nd try to keep the time by using os.path.split())

            for graph_num in nb_graphs_used:

                suffix_graph  = suffix + "_graphs_"+str(graph_num)
                dict_accuracy = {
                    "mSync": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, os.path.join("results_sub_graphs","X_mSync"+suffix_graph), False, sub_nb_graphs=graph_num, use_precision=use_precision),
                    "mALS": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, os.path.join("results_sub_graphs","X_mALS"+suffix_graph), False, sub_nb_graphs=graph_num, use_precision=use_precision),
                    "cao_cst_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, os.path.join("results_sub_graphs","X_cao_cst_o"+suffix_graph), False, sub_nb_graphs=graph_num, use_precision=use_precision)
                }

                dict_to_be_saved[graph_num] = dict_accuracy
            
            
        

    # If we look at mixed methodologies
    elif method == "mix":

        # initialize dict to be save
        dict_to_be_saved = {"parameter_correspondence" : dict_parameters_correspondence}

        for i_mix in range(10,101,10):

            suffix = "_mix_"+str(i_mix)

            start = time.time()
            dict_accuracy = {
                "mSync": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, os.path.join("results_mix","X_mSync"+suffix), False, use_precision=use_precision),
                "mALS": get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, os.path.join("results_mix","X_mALS"+suffix), False, use_precision=use_precision),
                "cao_cst_o":  get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, os.path.join("results_mix","X_cao_cst_o"+suffix), False, use_precision=use_precision)
            }
            print("get accuracy took ",(time.time() -start)* 1,"seconds")

            start = time.time()
            dict_distance = {
                "mSync": get_distance_mix_original(path_to_read, dict_parameters_correspondence, name_result_file="X_mSync", mix_rate = i_mix),
                "mALS": get_distance_mix_original(path_to_read, dict_parameters_correspondence, name_result_file="X_mALS", mix_rate = i_mix),
                "cao_cst_o":  get_distance_mix_original(path_to_read, dict_parameters_correspondence, name_result_file="X_cao_cst_o", mix_rate = i_mix),
            }
            print("get distances took ",(time.time() -start)* 1,"seconds")

            dict_to_be_saved[i_mix] = {"accuracy":dict_accuracy, "distance_to_original":dict_distance}
                
                                       
    

    # Save the dict in the given directory
    pickle_out = open(path_to_write,"wb")
    pickle.dump(dict_to_be_saved, pickle_out)
    pickle_out.close()
    


####################

if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Generate the result tensors from the algorithms that have ben ran on the simulated graphs")
    parser.add_argument("path_to_read", help="path where the folders contains the graphs and the algorithms results")
    parser.add_argument("path_to_write", help="path where to write the results")
    parser.add_argument("--method_pairwise", help="The method used to generate the original full pairwise matrix (KerGM or good_guess)", default="KerGM")
    parser.add_argument("--use_subgraphs", help="Whether or not the result of multigraph matching have been calculating using different numbers of graphs", default=0, type=int)
    parser.add_argument("--use_precision", help="Whether or not the loss measure used is the recall or the precision. Only important for partial matching algorithms", default=0, type=int)
    parser.add_argument("--use_F1", help="Whether or not the loss measure used is the F1 or not. Only important for partial matching algorithms or cases with no outliers", default=0, type=int)
    args = parser.parse_args()

    
    path_to_read = args.path_to_read
    path_to_write = args.path_to_write
    method_pairwise = args.method_pairwise
    use_subgraphs = args.use_subgraphs
    use_precision = args.use_precision
    use_F1 = args.use_F1

    possible_methods = ["KerGM", "good_guess", "mix"]
    if method_pairwise in possible_methods:
        get_and_write_all_results(path_to_read, path_to_write, method_pairwise, use_subgraphs, use_precision, use_F1)
    else:
        print("wong method name, it is not possible to continue. The available methods are KerGM, good_guess or mix")

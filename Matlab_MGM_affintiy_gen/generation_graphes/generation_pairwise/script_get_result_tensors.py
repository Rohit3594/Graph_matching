''' This scripts aims at generating n simulated sulcal-pits graph
    Author : Nathan Buskulic
    Creation date : March 11th 2020  
'''

import numpy as np
import scipy.io as sio
import pickle
import os
import argparse



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




def get_accuracy_permutation(target_permutation, given_permutation, use_precision=False):
    """ Return the accuracy of a given permutation knowing the ground truth permutation
        - ground_truth_permutation is a list where the index corresponds to the node in the first graph
        and the value at this index to the permuted node in the second graph
        - given_permutation is a n*n matrix where lines corresponds to node of the first graph
        and columns to the node of the second graph. A 1 in that matrix means that the two nodes are permuted
    """
    
    count_ok = 0
    for original_node, corresponding_node in enumerate(target_permutation):
        if given_permutation[original_node, corresponding_node] == 1:
            count_ok += 1

    # If we want the recall measure
    if not use_precision:
        count_total = len(target_permutation)
    else:
        count_total = np.nonzero(given_permutation)[0].shape[0]
        
    return np.clip(count_ok / count_total, 0, 1)



def get_result_tensor_for_given_algorithm(path_to_folder, param_correspondence_dict, name_of_result_file="X_kergm", transposed=False, use_precision=False):
    """Go through all folders and build a 3D tensor that holds
    the accuracy metric for all set of parameters for a given algorithm.
    
    Arguments:
        path_to_folder: The path where the results have been calculated
        param_correspondence_dict: The result of the get_noise_and_outliers_params
    function that give an integer correspondence to each parameter value
        name_of_result_file: The file name to load in each folder (each one
    corresponds to a given algorithm)
    """
    
    # We need to find how many runs were used so we go though the first folder
    nbRuns = len(os.listdir(os.path.join(path_to_folder, os.listdir(path_to_folder)[0])))
    
    # We initialise the final tensor that hold the results
    result_tensor_shape = [len(param_correspondence_dict[param_name]) for param_name in param_correspondence_dict]\
                          + [nbRuns]
    result_tensor = np.zeros(result_tensor_shape)
    
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
            
            # load the algorithm result
            algorithm_res = sio.loadmat(os.path.join(path_run_folder,name_of_result_file+".mat"))["X"]
            
            # transpose it if necessary
            if transposed:
                algorithm_res = np.transpose(algorithm_res)
            
            # get the accuracy result
            accuracy = get_accuracy_permutation(ground_truth, algorithm_res, use_precision)
            
            # add the result to the tensor
            result_tensor[param_1_integer, param_2_integer, run_i] = accuracy
            
    return result_tensor


def get_and_write_all_results(path_to_read, path_to_write, use_precision):
    """ Compute the result tensor for each algorithm and write
        the results in a given directory 
    """

    #load the dictionary of parameters integer correspondence
    dict_parameters_correspondence = get_noise_and_outliers_params(path_to_read)

    # compute the result of all the given algorithms
    transposed = True
    kergm_res = get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_kergm", False, use_precision)
    ipf_res = get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_ipf", transposed, use_precision)
    rrwm_res = get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_rrwm", transposed, use_precision)
    smac_res = get_result_tensor_for_given_algorithm(path_to_read, dict_parameters_correspondence, "X_smac", transposed, use_precision)

    # Create the final dict to be saved
    dict_to_be_saved = {"parameter_correspondence" : dict_parameters_correspondence,
                        "kergm": kergm_res,
                        "ipf": ipf_res,
                        "rrwm": rrwm_res,
                        "smac": smac_res 
    }

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
    parser.add_argument("--use_precision", help="Wether or not to use precision measure or recall measure (int 0 or 1)", default=0, type=int)
    args = parser.parse_args()

    
    path_to_read = args.path_to_read
    path_to_write = args.path_to_write
    use_precision = args.use_precision

    get_and_write_all_results(path_to_read, path_to_write, use_precision)


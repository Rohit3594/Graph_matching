import os
import sys
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_generation.load_graphs_and_create_metadata import dataset_metadata
from graph_matching_tools.metrics import matching
import matplotlib.pyplot as plt
import scipy.io as sio


path_to_graph_folder = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/simu_test_single_noise/'
path_to_dummy_graph_folder = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/simu_with_dummy/'

def insert_at(arr, output_size, indices):
    """
    Insert zeros at specific indices over whole dimensions, e.g. rows and/or columns and/or channels.
    You need to specify indices for each dimension, or leave a dimension untouched by specifying
    `...` for it. The following assertion should hold:

            `assert len(output_size) == len(indices) == len(arr.shape)`

    :param arr: The array to insert zeros into
    :param output_size: The size of the array after insertion is completed
    :param indices: The indices where zeros should be inserted, per dimension. For each dimension, you can
                specify: - an int
                         - a tuple of ints
                         - a generator yielding ints (such as `range`)
                         - Ellipsis (=...)
    :return: An array of shape `output_size` with the content of arr and zeros inserted at the given indices.
"""
    # assert len(output_size) == len(indices) == len(arr.shape)
    result = np.zeros(output_size)
    existing_indices = [np.setdiff1d(np.arange(axis_size), axis_indices,assume_unique=True)
                        for axis_size, axis_indices in zip(output_size, indices)]
    result[np.ix_(*existing_indices)] = arr
    return result


def get_output_size( input_size, len_idx):
    """
    :param input_size: INT, for exemple just 5
    :param len_idx:
    :return:
    """
    print(input_size)
    print(len_idx)
    ouput_size = input_size + len_idx
    return  ouput_size





if __name__ == '__main__':

	trials = np.sort(os.listdir(path_to_graph_folder))

	scores = {100:[],400:[],700:[],1000:[],1300:[]}

	for trial in trials:
	    print('trial: ', trial)
	    
	    all_files = os.listdir(path_to_graph_folder+trial)
	    
	    for folder in all_files:
	        
	        if os.path.isdir(path_to_graph_folder+trial+'/'+ folder):
	            
	            print('Noise folder: ',folder)
	            
	            path_to_graphs = path_to_graph_folder + '/' + trial + '/' + folder+'/graphs/'
	            path_to_dummy_graphs = path_to_dummy_graph_folder + '/' + trial +'/' + folder + '/0/graphs/'
	         
	            
	            noise = folder.split(',')[0].split('_')[1]
	            
	            graph_meta = dataset_metadata(path_to_graphs, path_to_groundtruth_ref)
	            
	            all_dummy_graphs = [nx.read_gpickle(path_to_dummy_graphs+'/'+g) for g in np.sort(os.listdir(path_to_dummy_graphs))]
	            
	            sizes_dummy = [nx.number_of_nodes(g) for g in all_dummy_graphs]
	            
	            print('sizes dummy: ', sum(sizes_dummy))
	                       
	            X_kmeans = sio.loadmat(path_to_graph_folder + '/' + trial + '/' + folder +'/X_kmeans.mat')['full_assignment_mat']   
	            dummy_mask = [list(nx.get_node_attributes(graph,'is_dummy').values()) for graph in all_dummy_graphs]
	            dummy_mask = sum(dummy_mask,[])
	            dummy_indexes = [i for i in range(len(dummy_mask)) if dummy_mask[i]==True] 
	            
	            print('X kmeans shape before dummy: ', X_kmeans.shape) 
	            
	            X_kmeans = insert_at(X_kmeans, (sum(sizes_dummy), sum(sizes_dummy)), (dummy_indexes, dummy_indexes))
	            
	            print('X kmeans shape after dummy: ',X_kmeans.shape)
	            
	            kmeans_X = {}
	            kmeans_X['full_assignment_mat'] = X_kmeans
	            
	            sio.savemat(path_to_graph_folder + '/' + trial + '/' + folder + '/X_kmeans_dummy.mat',kmeans_X)
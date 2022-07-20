import networkx as nx
import numpy as np
import scipy.io as sio
import os
from multiprocessing import Pool
import argparse
import pickle as pickle


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
		#print("problème")
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
		if not graph_1.nodes[node]["is_dummy"]:
			full_coordinates.append(graph_1.nodes[node]["coord"])
	graph_1_gamma_coord = compute_median_distances_all_pair(full_coordinates, "geodesic")

	# we get all the coordinates for graph_2
	full_coordinates = []
	for node in graph_2.nodes:
		if not graph_2.nodes[node]["is_dummy"]:
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

                    # Check that there s no dummy nodes
                    dummies = [graph_1.nodes[node_a]["is_dummy"],
                               graph_1.nodes[node_b]["is_dummy"],
                               graph_2.nodes[node_i]["is_dummy"],
                               graph_2.nodes[node_j]["is_dummy"]]
                    
                               
                    if not True in dummies:
                    
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


def load_generate_and_save_affinity_and_incidence_for_pair(path_to_folder, graph_nb_1, graph_nb_2, kernel_args, cpt_full_matrix=False):
	""" Generate the affinity and incidences matrix and save them
		in a given repositery
	"""

	print(path_to_folder, graph_nb_1, graph_nb_2)
	
	# get the two graphs
	graph_1 = pickle.load(open(os.path.join(path_to_folder,"graphs","graph_"+str(graph_nb_1)+".gpickle"),'rb'))
	graph_2 = pickle.load(open(os.path.join(path_to_folder,"graphs","graph_"+str(graph_nb_2)+".gpickle"),'rb'))
	#graph_1 = nx.read_gpickle(os.path.join(path_to_folder, "graphs", "graph_"+str(graph_nb_1)+".gpickle"))
	#graph_2 = nx.read_gpickle(os.path.join(path_to_folder, "graphs", "graph_"+str(graph_nb_2)+".gpickle"))

	# if the kernel is gaussian get the gamma value for the coordinate
	# and the geodesic distance
	if kernel_args[0] == "gaussian" and kernel_args[1]["gaussian_gamma"] == 0:
		gamma_coord, gamma_geodesic = compute_heuristic_gamma(graph_1, graph_2)
		print("gamma coord:",gamma_coord,"gamma geo:", gamma_geodesic)
		kernel_args[1]["gaussian_gamma_coord"] = gamma_coord
		kernel_args[1]["gaussian_gamma_geodesic"] = gamma_geodesic


	
	full_affinity_matrix = full_affinity(graph_1, graph_2, kernel_args)	

	dict_affinity_to_save = {"full_affinity":full_affinity_matrix}


	# Save everything in an appropriate matlab format
	sio.savemat(os.path.join(path_to_folder, "affinity_full", "full_affinity_"+str(graph_nb_1)+"_"+str(graph_nb_2)+".mat"), dict_affinity_to_save, do_compression=True)




def generate_and_save_all_affinity_and_incidence_in_path(path_to_folder, kernel_args, cpt_full_matrix=False, nb_workers=4):
	""" Go through all folders and subfolders to load graphs and
		generate the correponding affinity and incidence matrices.
		This process is done using subprocesses to increase the
		computation time
	"""

	# get all the informations to send to the processes
	list_arguments = []
	for file_name in os.listdir(path_to_folder):
		long_file_name = os.path.join(path_to_folder, file_name)
		print(long_file_name)
		if os.path.isdir(long_file_name):
			for sub_folder in os.listdir(long_file_name):
				full_folder_name = os.path.join(long_file_name,sub_folder)

				# Create the directory that will hold the affinity results
				if not os.path.isdir(os.path.join(full_folder_name,"affinity_full")):
					os.mkdir(os.path.join(full_folder_name,"affinity_full"))

				# get the number of graph
				files = os.listdir(os.path.join(full_folder_name, "graphs"))
				nb_tot_graphs = 0
				for fil in files:
					if 'graph' in fil:
						nb_tot_graphs += 1
				print('nb_tot_graphs=', nb_tot_graphs)
				# For each pair of graphs
				for i_graph in range(nb_tot_graphs):
					for j_graph in range(i_graph+1, nb_tot_graphs):
						
						list_arguments.append((full_folder_name,"{:05d}".format(i_graph),"{:05d}".format(j_graph), kernel_args, cpt_full_matrix))


	# launch the processes
	with Pool(processes=nb_workers) as pool:

		pool.starmap(load_generate_and_save_affinity_and_incidence_for_pair, list_arguments)


if __name__ == "__main__":

	# We parse the argument from command line
	parser = argparse.ArgumentParser(description="Generate the affinity and incidence matrices given a folder structure generated through the script to generate n graphs")
	parser.add_argument("path_to_folder", help="path where the folders contains the graphs")
	parser.add_argument("--nb_workers", help="number of processes to launch", default=4, type=int)
	parser.add_argument("--cpt_full_matrix", help="Decide if we the file should include full affinity matrices or just the smaller one for KerGM (0=False, 1=True)", type=int, default=1)
	parser.add_argument("--kernel_type", help="kernel type, only gaussian right now", default="gaussian")
	parser.add_argument("--gaussian_gamma", help="gamma value for the gaussian kernel", default=0, type=float)
	args = parser.parse_args()


	
	#path_to_folder = "/hpc/meca/users/rohit/stage_nathan/generation_graphes/generation_multi/test_for_pairwise"  #args.path_to_folder
	path_to_folder = args.path_to_folder
	cpt_full_matrix = bool(args.cpt_full_matrix)
	nb_workers = args.nb_workers
	gaussian_gamma = args.gaussian_gamma
	kernel_type = args.kernel_type
	
	# We define the kernel arguments to be used
	kernel_args = (kernel_type, {"gaussian_gamma":gaussian_gamma})
	
	generate_and_save_all_affinity_and_incidence_in_path(path_to_folder,
														 kernel_args,
														 cpt_full_matrix=cpt_full_matrix,
														 nb_workers=nb_workers)
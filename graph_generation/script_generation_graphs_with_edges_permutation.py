import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import argparse
import numpy as np
import networkx as nx
import slam.plot as splt
import slam.topology as stop
import slam.generate_parametric_surfaces as sgps
import trimesh
import os
import tools.graph_processing as gp
from sphere import *
from tqdm.auto import tqdm,trange
import random



def generate_reference_graph(nb_vertices, radius):
    # Generate random sampling
    sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_vertices, radius=radius)

    sphere_random_sampling = tri_from_hull(sphere_random_sampling)  # Computing convex hull (adding edges)

    adja = stop.edges_to_adjacency_matrix(sphere_random_sampling)
    graph = nx.from_numpy_matrix(adja.todense())
    # Create dictionnary that will hold the attributes of each node
    node_attribute_dict = {}
    for node in graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(sphere_random_sampling.vertices[node])}

    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)

    # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in graph.edges:
        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = gp.compute_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(graph, edge_attribute_dict)

    return graph


def tri_from_hull(vertices):
    """
	compute faces from vertices using trimesh convex hull
	:param vertices: (n, 3) float
	:return:
	"""
    mesh = trimesh.Trimesh(vertices=vertices, process=False)
    return mesh.convex_hull


def edge_len_threshold(graph,thr): # Adds a percentage of edges 
    
    edge_to_add = random.sample(list(graph.edges),round(len(graph.edges)*thr))

    return edge_to_add


def generate_sphere_random_sampling(vertex_number=100, radius=1.0):
    """
	generate a sphere with random sampling
	:param vertex_number: number of vertices in the output spherical mesh
	:param radius: radius of the output sphere
	:return:
	"""
    coords = np.zeros((vertex_number, 3))
    for i in range(vertex_number):
        M = np.random.normal(size=(3, 3))
        Q, R = np.linalg.qr(M)
        coords[i, :] = Q[:, 0].transpose() * np.sign(R[0, 0])
    if radius != 1:
        coords = radius * coords
    return coords


def generate_noisy_graph(original_graph, nb_vertices, nb_outliers, sigma_noise_nodes=1, sigma_noise_edges=1,
                         radius=100):
    # Perturbate the coordinates

    noisy_coord = []
    key = []
    value = []

    for index in range(nb_vertices):
        # Sampling from Von Mises - Fisher distribution
        original_coord = original_graph.nodes[index]["coord"]
        mean_original = original_coord / np.linalg.norm(original_coord)  # convert to mean unit vector.
        noisy_coordinate = Sphere().sample(1, distribution='vMF', mu=mean_original,
                                           kappa=sigma_noise_nodes).sample[0]

        noisy_coordinate = noisy_coordinate * np.linalg.norm(original_coord) # rescale to original size.
        # print(noisy_coordinate)
        noisy_coord.append(noisy_coordinate)

    # Add Outliers
    sphere_random_sampling = []
    if nb_outliers > 0:
        #print("nb_outliers: ", nb_outliers)
        sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_outliers, radius=radius)
        # merge pertubated and outlier coordinates to add edges 
        all_coord = noisy_coord + list(sphere_random_sampling)
    else:
        all_coord = noisy_coord


    noisy_graph = nx.Graph()

    compute_noisy_edges = tri_from_hull(all_coord)  # take all peturbated coord and comp conv hull.
    adja = stop.edges_to_adjacency_matrix(compute_noisy_edges)  # compute the new adjacency mat.

    noisy_graph = nx.from_numpy_matrix(adja.todense())

    node_attribute_dict = {}
    for node in noisy_graph.nodes():
        node_attribute_dict[node] = {"coord": np.array(compute_noisy_edges.vertices[node])}

    nx.set_node_attributes(noisy_graph, node_attribute_dict)

    nx.set_edge_attributes(noisy_graph, 1.0, name="weight")

    edge_attribute_dict = {}
    id_counter = 0  # useful for affinity matrix caculation
    for edge in noisy_graph.edges:
        # We calculate the geodesic distance
        end_a = noisy_graph.nodes()[edge[0]]["coord"]
        end_b = noisy_graph.nodes()[edge[1]]["coord"]
        geodesic_dist = gp.compute_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id": id_counter}
        id_counter += 1

    # add the edge attributes to the graph
    nx.set_edge_attributes(noisy_graph, edge_attribute_dict)

    # Extracting the ground-truth correspondence

    ground_truth_permutation = []

    for ar1 in noisy_coord:
        for i in range(len(noisy_graph.nodes)):

            if np.mean(ar1) == np.mean(noisy_graph.nodes[i]['coord']):    
                if i >=nb_vertices:
                    key.append(i)
                ground_truth_permutation.append(i)
                break

    #print("len ground_truth_permutation: ", len(ground_truth_permutation))
    #print("len noisy_coord : ", len(noisy_coord))

    for outlier in sphere_random_sampling:
        for i in range(len(noisy_graph.nodes)):
            if np.mean(noisy_graph.nodes[i]['coord']) == np.mean(outlier):
                if i<nb_vertices:
                    value.append(i)


    if nb_outliers > 0 and len(key)!=0:
        index = 0
        for j in range(len(ground_truth_permutation)):
            if ground_truth_permutation[j] == key[index]:
                ground_truth_permutation[j] = value[index]
                index+=1
                if index == len(key):
                    break

        key = key + value
        value = value + key

        mapping = dict(zip(key,value))
        #print("mapping :",mapping)
        #print("number of nodes in graphs: ", len(noisy_graph.nodes))
        noisy_graph = nx.relabel_nodes(noisy_graph, mapping)


    # Remove 10% of random edges
    edge_to_remove = edge_len_threshold(noisy_graph, 0.10)
    noisy_graph.remove_edges_from(edge_to_remove)

    noisy_graph.remove_edges_from(nx.selfloop_edges(noisy_graph))


    return ground_truth_permutation, noisy_graph


def get_nearest_neighbors(original_coordinates, list_neighbors, radius, nb_to_take=10):
    ''' Return the nb_to_take nearest neighbors (in term of geodesic distance) given a set
		of original coordinates and a list of tuples where the first term is the label 
		of the node and the second the associated coordinates
	'''

    # We create the list of distances and sort it
    distances = [(i, gp.compute_geodesic_distance_sphere(original_coordinates, current_coordinates, radius)) for
                 i, current_coordinates in list_neighbors]
    distances.sort(key=lambda x: x[1])

    return distances[:nb_to_take]


def add_integer_id_to_edges(graph):
    """ Given a graph, add an attribute "id" to each edge that is a unique integer id"""

    dict_attributes = {}
    id_counter = 0
    for edge in graph.edges:
        dict_attributes[edge] = {"id": id_counter}
        id_counter += 1
    nx.set_edge_attributes(graph, dict_attributes)


def mean_edge_len(G):
    all_geo = [z['geodesic_distance'] for x, y, z in list(G.edges.data())]
    #mean_geo = np.array(all_geo).mean()
    # std = np.std(all_geo)

    return all_geo


def get_in_between_perm_matrix(perm_mat_1, perm_mat_2):
    """
	Given two permutation from noisy graphs to a reference graph,
	Return the permutation matrix to go from one graph to the other
	"""
    result_perm = np.zeros((perm_mat_1.shape[0],), dtype=int)

    for node_reference, node_noisy_1 in enumerate(perm_mat_1):
        # get the corresponding node in the second graph
        node_noisy_2 = perm_mat_2[node_reference]

        # Fill the result
        result_perm[node_noisy_1] = node_noisy_2

    return result_perm


def generate_graph_family(nb_sample_graphs, nb_graphs, nb_vertices, radius, nb_outliers, ref_graph, noise_node=1, noise_edge=1,
                          nb_neighbors_to_consider=10):
    """
	Generate n noisy graphs from a reference graph alongside the 
	ground truth permutation matrices.
	"""
    # Generate the reference graph
    reference_graph = ref_graph

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []
    list_ground_truth = []

    # We generate the n noisy graphs
    print("Generating graphs..")

    for c_graph in tqdm(range(nb_sample_graphs)):

        ground_truth, noisy_graph = generate_noisy_graph(reference_graph, nb_vertices, nb_outliers, noise_node,
                                                         noise_edge)

        # Add outliers
        # add_outliers(noisy_graph, nb_outliers, nb_neighbors_to_consider, radius)

        if nx.is_connected(noisy_graph) == False:
            continue

        if nx.is_connected(noisy_graph) == False:
            print("Found disconnected components..!!")

        # Add id to edge
        add_integer_id_to_edges(noisy_graph)

        # Save the graph
        list_noisy_graphs.append(noisy_graph)

        # Save all ground-truth for later selecting the selected graphs
        list_ground_truth.append(ground_truth)



    min_geo = []
    selected_graphs = []
    selected_ground_truth = []

    for graphs,gt in zip(list_noisy_graphs,list_ground_truth):
        z = mean_edge_len(graphs)
    
        if min(z) > 7.0:
            selected_graphs.append(graphs) # select the noisy graph.
            selected_ground_truth.append(gt) # and its corresponding ground-truth.
            min_geo.append(min(z))


    sorted_zipped_lists = zip(min_geo, selected_graphs, selected_ground_truth)
    sorted_zipped_lists = sorted(sorted_zipped_lists,reverse = True)

    sorted_graphs = []
    sorted_ground_truth = []

    for l,m,n in sorted_zipped_lists:
        sorted_graphs.append(m)
        sorted_ground_truth.append(n)


    print("Verifying len of sorted_graphs,sorted_ground_truth,min_geo(should be equal):",len(sorted_graphs),len(sorted_ground_truth),len(min_geo))
 

    # Initialise permutation matrices to reference graph
    ground_truth_perm_to_ref = np.zeros((nb_graphs, nb_vertices), dtype=int)
    ground_truth_perm = np.zeros((nb_graphs, nb_graphs, nb_vertices), dtype=int)


    # Save the ground truth permutation
    count = 0
    for ground_truth in sorted_ground_truth[:nb_graphs]: # Select the nb_graphs with largest min-geo distance
        ground_truth_perm_to_ref[count, :] = ground_truth
        count +=1 


    # We generate the ground_truth permutation between graphs
    print("Groundtruth Labeling..")
    for i_graph in tqdm(range(nb_graphs)):
        for j_graph in range(nb_graphs):
            ground_truth_perm[i_graph, j_graph, :] = \
                get_in_between_perm_matrix(ground_truth_perm_to_ref[i_graph, :], ground_truth_perm_to_ref[j_graph, :])

    return sorted_graphs[:nb_graphs] , ground_truth_perm


def generate_n_graph_family_and_save(path_to_write, nb_runs, nb_ref_graph, nb_sample_graphs,nb_graphs, nb_vertices,
                                     radius, list_noise, list_outliers, nb_neighbors_to_consider=10, save_reference=0):
    ''' Generate n family of graphs for each couple (noise, outliers). The graphs are saved
		in a folder structure at the point path_to_write
	'''

    # check if the path given is a folder otherwise create one
    if not os.path.isdir(path_to_write):
        os.mkdir(path_to_write)

    # generate n families of graphs
    for i_graph in range(nb_runs):

        # Select the ref graph with highest mean geo distance
        print("Generating reference_graph..")
        for i in tqdm(range(nb_ref_graph)):
            reference_graph = generate_reference_graph(nb_vertices, radius)
            all_geo = mean_edge_len(reference_graph)

            if i == 0:
                min_geo = min(all_geo)

            else:

                if min(all_geo) > min_geo:
                    min_geo = min(all_geo)
                    reference_graph_max = reference_graph

                else:
                    pass

        if save_reference:
            print("Selected reference graph with min_geo: ",min_geo)
            trial_path = os.path.join(path_to_write, str(i_graph))  # for each trial
            if not os.path.isdir(trial_path):
                os.mkdir(trial_path)
            nx.write_gpickle(reference_graph_max, os.path.join(trial_path, "reference_" + str(i_graph) + ".gpickle"))

        for noise in list_noise:
            for outliers in list_outliers:

                folder_name = "noise_" + str(noise) + ",outliers_" + str(outliers)
                path_parameters_folder = os.path.join(trial_path, folder_name)

                if not os.path.isdir(path_parameters_folder):
                    os.mkdir(path_parameters_folder)
                    os.mkdir(os.path.join(path_parameters_folder, "graphs"))

                list_graphs,ground_truth_perm  = generate_graph_family(nb_sample_graphs= nb_sample_graphs,nb_graphs=nb_graphs,
                                                                       nb_vertices=nb_vertices,
                                                                       radius=radius,
                                                                       nb_outliers=outliers,
                                                                       ref_graph=reference_graph_max,
                                                                       noise_node=noise,
                                                                       noise_edge=noise,
                                                                       nb_neighbors_to_consider=nb_neighbors_to_consider)

                for i_family, graph_family in enumerate(list_graphs):
                    nx.write_gpickle(graph_family, os.path.join(path_parameters_folder, "graphs",
                                                                "graph_" + str(i_family) + ".gpickle"))

                np.save(os.path.join(path_parameters_folder, "ground_truth"), ground_truth_perm)



if __name__ == '__main__':
    path_to_write = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/simu_graph/final_new_simu/other_9_trials/'

    nb_runs = 1
    nb_sample_graphs = 5000 #  # of graphs to generate before selecting the NN graphs with highest geodesic distance.
    nb_graphs = 134 # nb of graphs to generate
    nb_vertices = 72  #72 based on Kaltenmark, MEDIA, 2020
    min_noise = 200
    max_noise = 1600
    step_noise = 600
    min_outliers = 0
    max_outliers = 24
    step_outliers = 6
    save_reference = 1
    nb_ref_graph = 5000
    radius = 100


    list_noise = np.arange(min_noise, max_noise, step_noise)
    list_outliers = np.array(list(range(min_outliers, max_outliers, step_outliers)))
    nb_neighbors_to_consider_outliers = 10

    # call the generation procedure
    generate_n_graph_family_and_save(path_to_write=path_to_write,
                                     nb_runs=nb_runs,
                                     nb_ref_graph=nb_ref_graph,
                                     nb_sample_graphs=nb_sample_graphs,
                                     nb_graphs = nb_graphs,
                                     nb_vertices=nb_vertices,
                                     radius=radius,
                                     list_noise=list_noise,
                                     list_outliers=list_outliers,
                                     nb_neighbors_to_consider=nb_neighbors_to_consider_outliers,
                                     save_reference=save_reference)

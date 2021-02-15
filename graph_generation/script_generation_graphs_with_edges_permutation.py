''' This scripts aims at generating n simulated sulcal-pits graph
    Author : Nathan Buskulic
    Creation date : March 11th 2020  
'''
import argparse
import numpy as np
import networkx as nx
import slam.plot as splt
import slam.topology as stop
import slam.generate_parametric_surfaces as sgps
import trimesh
import os
import tools.graph_processing as gp

def generate_reference_graph(nb_vertices, radius):
    
    # Generate random sampling
    sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_vertices, radius=radius)

    sphere_random_sampling = tri_from_hull(sphere_random_sampling) # Computing convex hull (adding edges)


    adja = stop.edges_to_adjacency_matrix(sphere_random_sampling)
    graph = nx.from_numpy_matrix(adja.todense())
    # Create dictionnary that will hold the attributes of each node
    node_attribute_dict = {}
    for node in graph.nodes():
        node_attribute_dict[node] = {"coord" : np.array(sphere_random_sampling.vertices[node])}

    # add the node attributes to the graph
    nx.set_node_attributes(graph, node_attribute_dict)


    # We add a default weight on each edge of 1
    nx.set_edge_attributes(graph, 1.0, name="weight")

    # We add a geodesic distance between the two ends of an edge
    edge_attribute_dict = {}
    id_counter = 0 # useful for affinity matrix caculation
    for edge in graph.edges:

        # We calculate the geodesic distance
        end_a = graph.nodes()[edge[0]]["coord"]
        end_b = graph.nodes()[edge[1]]["coord"]
        geodesic_dist = gp.compute_geodesic_distance_sphere(end_a, end_b, radius)

        # add the information in the dictionnary
        edge_attribute_dict[edge] = {"geodesic_distance": geodesic_dist, "id":id_counter}
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



def generate_noisy_graph(original_graph, nb_vertices, sigma_noise_nodes = 1, sigma_noise_edges = 1, radius = 100):
    
    # generate ground_truth permutation
    ground_truth_permutation = np.random.permutation(nb_vertices)
    
    # create a new graph
    noisy_graph = nx.Graph()

    # add the nodes (not very clean but it works fine and run in no time)
    for node_to_add in range(len(ground_truth_permutation)):
        for original_node, current_node in enumerate(ground_truth_permutation):
            if current_node == node_to_add:
                noisy_coordinate = original_graph.nodes[original_node]["coord"] + \
                    np.random.multivariate_normal(np.zeros(3), np.eye(3) * sigma_noise_nodes)

                # We project on the sphere
                noisy_coordinate = noisy_coordinate / np.linalg.norm(noisy_coordinate) * radius
                noisy_graph.add_node(node_to_add, coord = noisy_coordinate)

    # add the necessary nodes
    # for original_node, corresponding_node in enumerate(ground_truth_permutation):
    #     noisy_coordinate = original_graph.nodes[original_node]["coord"] + \
    #                        np.random.multivariate_normal(np.zeros(3), np.eye(3) * sigma_noise_nodes)
    #     # On projete ce point sur la sphère
    #     noisy_coordinate = noisy_coordinate / np.linalg.norm(noisy_coordinate) * radius
    #     noisy_graph.add_node(corresponding_node, coord = noisy_coordinate)
        
    compute_noisy_edges = tri_from_hull(list(nx.get_node_attributes(noisy_graph,'coord').values())) # take all peturbated coord and comp conv hull.
    adja = stop.edges_to_adjacency_matrix(compute_noisy_edges) # compute the new adjacency mat
    edge_list = [tuple((u,v)) for u,v in zip(adja.nonzero()[0],adja.nonzero()[1])] # convert to edge list to add edge attributes.


    # add the edges
    for edge in edge_list:

        # get the original and corresponding ends
        #end_a_corresponding, end_b_corresponding = ground_truth_permutation[edge[0]], ground_truth_permutation[edge[1]]
        coordinate_a, coordinate_b = noisy_graph.nodes[edge[0]]["coord"], noisy_graph.nodes[edge[1]]["coord"]

        # calculate noisy geodesic distance
        noisy_geodesic_dist = gp.compute_geodesic_distance_sphere(coordinate_a, coordinate_b, radius)

        # Add the new edge to the graph
        noisy_graph.add_edge(edge[0],edge[1], weight = 1.0, geodesic_distance = noisy_geodesic_dist)
    
    return ground_truth_permutation, noisy_graph




def get_nearest_neighbors(original_coordinates, list_neighbors, radius, nb_to_take=10):
    ''' Return the nb_to_take nearest neighbors (in term of geodesic distance) given a set 
        of original coordinates and a list of tuples where the first term is the label 
        of the node and the second the associated coordinates
    '''
    
    # We create the list of distances and sort it
    distances = [(i, gp.compute_geodesic_distance_sphere(original_coordinates, current_coordinates, radius)) for i, current_coordinates in list_neighbors]
    distances.sort(key = lambda x: x[1])
    
    return distances[:nb_to_take]


def add_outliers(graph, nb_outliers, nb_neighbors_to_consider, radius):
    '''Add outlier to a graph such that the values of nodes and edges are plausible.
       The nodes are sampled from a sphere and the edges are chosen by randomly selecting
       nearest neighbors.
    '''
    
    # We save the mean degree of the graph since it is useul later (for the probability of choosing neighbors)
    mean_degree = np.array(graph.degree())[:,1].mean()
    
    # We save the original number of nodes
    original_number_of_nodes = graph.number_of_nodes()
    
    # We take the outliers by sampling randomly on the sphere
    # in order to make sure that the operation is always possible we need to have at least 4 outliers
    sphere_random_sampling = generate_sphere_random_sampling(vertex_number=nb_outliers + 4, radius=radius)
    
    # we add the nodes to the graph
    for i in range(nb_outliers):
        node_label = original_number_of_nodes + i
        graph.add_node(node_label, coord = np.array(sphere_random_sampling.vertices[i]))
        
    # we add the edges
    for i in range(nb_outliers):
        
        # In order to add the edges we first create a list of nodes and
        # the associated coordinate.
        node_label = original_number_of_nodes + i
        list_neighbors = [(node, graph.nodes[node]["coord"]) for node in graph.nodes if node != node_label]
        
        # get the nearest neighbors
        nearest_neighbors = get_nearest_neighbors(graph.nodes[node_label]["coord"], \
                                                  list_neighbors, radius, nb_to_take=nb_neighbors_to_consider)
        
        # select a subset of neighbors based on the statistics obtain from the graph
        subset_of_neigbors = []
        probability_of_chosen = mean_degree / nb_neighbors_to_consider
        for node, distance in nearest_neighbors:
            if np.random.rand(1) < probability_of_chosen:
                subset_of_neigbors.append(node)
                
        # We add all the corresponding edges
        for connected_node in subset_of_neigbors:
            vec_a = graph.nodes[node_label]["coord"]
            vec_b = graph.nodes[connected_node]["coord"]
            graph.add_edge(node_label, connected_node , weight = 1.0,
                           geodesic_distance = gp.compute_geodesic_distance_sphere(vec_a, vec_b, radius))


def add_integer_id_to_edges(graph):
    """ Given a graph, add an attribute "id" to each edge that is a unique integer id"""

    dict_attributes={}
    id_counter = 0
    for edge in graph.edges:
        dict_attributes[edge] = {"id":id_counter}
        id_counter += 1
    nx.set_edge_attributes(graph, dict_attributes)




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

    
def generate_graph_family(nb_graphs, nb_vertices, radius, nb_outliers,ref_graph, noise_node=1, noise_edge=1, nb_neighbors_to_consider=10):
    """
    Generate n noisy graphs from a reference graph alongside the 
    ground truth permutation matrices.
    """
    
    # Generate the reference graph
    reference_graph = ref_graph

    # Initialise permutation matrices to reference graph
    ground_truth_perm_to_ref = np.zeros((nb_graphs, nb_vertices), dtype=int)
    ground_truth_perm = np.zeros((nb_graphs, nb_graphs, nb_vertices), dtype=int)

    # Initialise the list of noisy_graphs
    list_noisy_graphs = []

    # We generate the n noisy graphs
    for c_graph in range(nb_graphs):
        

        ground_truth, noisy_graph = generate_noisy_graph(reference_graph, nb_vertices, noise_node, noise_edge)

        # Add outliers
        add_outliers(noisy_graph, nb_outliers, nb_neighbors_to_consider, radius)
        
        # Add id to edge
        add_integer_id_to_edges(noisy_graph)
        
        # Save the ground truth permutation
        ground_truth_perm_to_ref[c_graph,:] = ground_truth

        # Save the graph
        list_noisy_graphs.append(noisy_graph)
        
    # We generate the ground_truth permutation  between graphs
    for i_graph in range(nb_graphs):
        for j_graph in range(nb_graphs):
            ground_truth_perm[i_graph, j_graph, :] = \
                get_in_between_perm_matrix(ground_truth_perm_to_ref[i_graph,:], ground_truth_perm_to_ref[j_graph,:])
            
    return ground_truth_perm, list_noisy_graphs


def generate_n_graph_family_and_save(path_to_write, nb_runs, nb_graphs, nb_vertices, radius, list_noise, list_outliers, nb_neighbors_to_consider=10, save_reference=0):
    ''' Generate n family of graphs for each couple (noise, outliers). The graphs are saved
        in a folder structure at the point path_to_write
    '''

    reference_graph = generate_reference_graph(nb_vertices, radius)


    if save_reference:
        path_reference_graph = path_to_write
        nx.write_gpickle(reference_graph, os.path.join(path_reference_graph, "reference.gpickle"))


    # check if the path given is a folder otherwise create one
    if not os.path.isdir(path_to_write):
        os.mkdir(path_to_write)

    # for each set of parameters
    for noise in list_noise:
        for outliers in list_outliers:

            folder_name = "noise_"+str(noise)+",outliers_"+str(outliers)
            path_parameters_folder = os.path.join(path_to_write,folder_name)
            print(path_parameters_folder)
            # If the folder does not exist create one
            if not os.path.isdir(path_parameters_folder):
                os.mkdir(path_parameters_folder)


            # generate n families of graphs
            for i_graph in range(nb_runs):

                ground_truth_perm, list_graphs = generate_graph_family(nb_graphs=nb_graphs, 
                                                                       nb_vertices=nb_vertices, 
                                                                       radius=radius, 
                                                                       nb_outliers=outliers,
                                                                       ref_graph = reference_graph, 
                                                                       noise_node=noise, 
                                                                       noise_edge=noise, 
                                                                       nb_neighbors_to_consider=nb_neighbors_to_consider)
                

                # save the graph in a folder i_graph
                path_graph_folder = os.path.join(path_parameters_folder,str(i_graph))
                if not os.path.isdir(path_graph_folder):
                    os.mkdir(path_graph_folder)
                    os.mkdir(os.path.join(path_graph_folder,"graphs"))

                for i_family, graph_family in enumerate(list_graphs):
                    nx.write_gpickle(graph_family, os.path.join(path_graph_folder,"graphs","graph_"+str(i_family)+".gpickle"))
                
                np.save(os.path.join(path_graph_folder, "ground_truth"), ground_truth_perm)
                

if __name__ == '__main__':

    # # We parse the argument from command line
    # parser = argparse.ArgumentParser(description="Generate simulated sulcal pits graphs based on two variables : noise and outliers")
    # parser.add_argument("path_to_write", help="path where the folders will be generated")
    # parser.add_argument("--nb_runs", help="number of families of graph to generate per set of parameters", default=5, type=int)
    # parser.add_argument("--nb_graphs", help="number of graph to create for set of parameters", default=5, type=int)
    # parser.add_argument("--nb_vertices", help="number of vertices to create the graph with", default=90, type=int)
    # parser.add_argument("--min_noise", help="minimum noise value", default=1, type=float)
    # parser.add_argument("--max_noise", help="maximum noise value", default=2,  type=float)
    # parser.add_argument("--step_noise", help="step size for noise values", default=0.1, type=float)
    # parser.add_argument("--min_outliers", help="minimum outliers value", default=1, type=int)
    # parser.add_argument("--max_outliers", help="maximum outliers value", default=10, type=int)
    # parser.add_argument("--step_outliers", help="step size for outliers values", default=1, type=int)
    # parser.add_argument("--save_reference", help="Wether to save the reference graph or not. Should not be used unless for visualisation purpose. (0 or 1)", default=0, type=int)
    # args = parser.parse_args()
    #
    #
    # path_to_write = args.path_to_write
    # nb_runs = args.nb_runs
    # nb_graphs = args.nb_graphs
    # nb_vertices = args.nb_vertices
    # min_noise = args.min_noise
    # max_noise = args.max_noise
    # step_noise = args.step_noise
    # min_outliers = args.min_outliers
    # max_outliers = args.max_outliers
    # step_outliers = args.step_outliers
    # save_reference = args.save_reference
    path_to_write = '../data/simu_graph'
    # path_to_write= '/Users/Nathan/stageINT/stage_nathan/generation_graphes/generation_multi/test_change_node_generation'
    # path_to_write='/hpc/meca/users/buskulic.n/stage_nathan/generation_graphes/generation_multi/test_for_pairwise'
    # #path_to_write = '/hpc/meca/users/auzias/ISBI2020_graph_matching/simu/generation_multi'
    nb_runs = 1
    nb_graphs = 130
    nb_vertices = 85
    min_noise = 100
    max_noise = 140
    step_noise = 10
    min_outliers = 0
    max_outliers = 5
    step_outliers = 5
    save_reference = 1
    # # We define the parameters used throughout the script
    # nb_vertices = nb_vertices
    radius = 100

    list_noise = np.arange(min_noise, max_noise, step_noise)
    list_outliers = np.array(list(range(min_outliers, max_outliers, step_outliers)))
    nb_neighbors_to_consider_outliers = 10

    # call the generation procedure
    generate_n_graph_family_and_save(path_to_write = path_to_write,
                                     nb_runs = nb_runs,
                                     nb_graphs = nb_graphs,
                                     nb_vertices = nb_vertices,
                                     radius = radius,
                                     list_noise = list_noise,
                                     list_outliers = list_outliers,
                                     nb_neighbors_to_consider = nb_neighbors_to_consider_outliers,
                                     save_reference = save_reference)


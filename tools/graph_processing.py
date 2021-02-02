import os
import numpy as np
import networkx as nx
import pickle


def load_graphs_in_list(path):
    """
    Return a list of graph loaded from the path
    """
    path_to_graphs_folder = os.path.join(path, "modified_graphs")
    list_graphs = []
    for i_graph in range(0, len(os.listdir(path_to_graphs_folder))):
        path_graph = os.path.join(path_to_graphs_folder, "graph_"+str(i_graph)+".gpickle")
        graph = nx.read_gpickle(path_graph)
        list_graphs.append(graph)

    return list_graphs


def graph_nodes_to_coords(graph, index_attribute, mesh):
    vert_indices = list(nx.get_node_attributes(graph, index_attribute).values())
    coords = np.array(mesh.vertices[vert_indices, :])
    return coords


def graph_nodes_attribute(graph, attribute):
    """
    get the 'attribute' node attribute from 'graph' as a numpy array
    :param graph: networkx graph object
    :param attribute: string, node attribute to be extracted
    :return: a numpy array where i'th element corresponds to the i'th node in the graph
    if 'attribute' is not a valid node attribute in graph, then the returned arry is empty
    """
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    if True in is_dummy:
        graph.remove_nodes_from(np.where(np.array(is_dummy) == True)[0])


def get_geodesic_distance_sphere(coord_a, coord_b, radius):
    '''
    Return the geodesic distance of two 3D vectors on a sphere
    '''
    return radius * np.arccos(np.clip(np.dot(coord_a, coord_b) / np.power(radius, 2), -1, 1))


def add_geodesic_distance_on_edges(graph):
    """
    Compute the geodesic distance represented by each edge
    and add it as attribute in the graph
    """

    # initialise the dict for atttributes on edges
    edges_attributes = {}

    # Fill the dictionnary with the geodesic_distance
    for edge in graph.edges:
        geodesic_distance = get_geodesic_distance_sphere(graph.nodes[edge[0]]["sphere_3dcoords"],
                                                         graph.nodes[edge[1]]["sphere_3dcoords"],
                                                         radius=100)

        edges_attributes[edge] = {"geodesic_distance": geodesic_distance}

    nx.set_edge_attributes(graph, edges_attributes)


def add_id_on_edges(graph):
    """
    Add an Id information on edge (integer)
    """

    # initialise the dict for atttributes on edges
    edges_attributes = {}

    # Fill the dictionnary with the geodesic_distance
    for i, edge in enumerate(graph.edges):
        edges_attributes[edge] = {"id": i}

    nx.set_edge_attributes(graph, edges_attributes)


def add_dummy_nodes(graph, nb_node_to_reach):
    """
    Add a given number of dummy nodes to the graph
    """
    for _ in range(graph.number_of_nodes(), nb_node_to_reach):
        graph.add_node(graph.number_of_nodes(), is_dummy=True)


def transform_3dcoords_attribute_into_ndarray(graph):
    """
    Transform the node attribute sphere_3dcoord from a list to a ndarray
    """
    # initialise the dict for atttributes on edges
    nodes_attributes = {}

    # Fill the dictionnary with the nd_array attribute
    for node in graph.nodes:
        nodes_attributes[node] = {"sphere_3dcoords": np.array(graph.nodes[node]["sphere_3dcoords"])}

    nx.set_node_attributes(graph, nodes_attributes)


def preprocess_graph(graph):
    """
    preprocessing of graphs
    :param graph:
    :return:
    """

    # transform the 3d attributes into ndarray
    transform_3dcoords_attribute_into_ndarray(graph)

    # Compute the geodesic distance for each node and add the id information
    add_geodesic_distance_on_edges(graph)

    # add ID identifier on edges
    add_id_on_edges(graph)

    # add the 'is_dummy' attribute to nodes, that will be used when manipulating dummy nodes later
    nx.set_node_attributes(graph, values=False, name="is_dummy")





###################################################################
# main function coded by Nathan to preprocess all real data graphs
###################################################################
def read_modify_and_write_graphs(path_to_folder):
    """
    Read a list of graph in a folder, add dummy nodes where it's
    necessary in order to have an equal number of nodes between all graphs
    and finally write the modified graphs in a new folder with a
    dictionnary to allow the correspondences.
    """

    # Initialise correspondence dictionary
    correspondence_dict = {}

    # initialise list of graphs
    graph_list = []

    # load all the graphs one after the other
    for graph_i, graph_file in enumerate([file_name for file_name in os.listdir(path_to_folder) if
                                          not os.path.isdir(os.path.join(path_to_folder, file_name))]):
        # load this graph
        graph = nx.read_gpickle(os.path.join(path_to_folder, graph_file))
        preprocess_graph(graph)
        graph_list.append(graph)  # add it to the list of graph
        correspondence_dict[graph_i] = {"name": graph_file}  # add the information about the name.


    # add the number of nodes information to the dict and find the max number of nodes
    max_nb_nodes = 0
    for i, graph in enumerate(graph_list):
        correspondence_dict[i]["nb_nodes"] = graph.number_of_nodes()
        if graph.number_of_nodes() > max_nb_nodes:
            max_nb_nodes = graph.number_of_nodes()

    # add the dummy nodes
    for graph in graph_list:
        add_dummy_nodes(graph, max_nb_nodes)

    # Create the new folder for the graphs
    new_folder_path = os.path.join(path_to_folder, "modified_graphs")
    if not os.path.isdir(new_folder_path):
        os.mkdir(new_folder_path)

    # Save the graphs in the new folder
    for i, graph in enumerate(graph_list):
        graph_path = os.path.join(new_folder_path, "graph_" + str(i) + ".gpickle")
        nx.write_gpickle(graph, graph_path)

    # Save the correspondence_dict
    pickle_out = open(os.path.join(path_to_folder, "correspondence_dict.pickle"), "wb")
    pickle.dump(correspondence_dict, pickle_out)
    pickle_out.close()



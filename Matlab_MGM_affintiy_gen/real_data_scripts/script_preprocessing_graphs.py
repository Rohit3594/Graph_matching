import os
import pickle
import numpy as np
import networkx as nx
import argparse


def get_geodesic_distance_sphere(coord_a, coord_b, radius):
    ''' 
    Return the geodesic distance of two 3D vectors on a sphere
    '''
    return radius * np.arccos(np.clip(np.dot(coord_a, coord_b) / np.power(radius,2),-1,1))

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
    for _ in range(graph.number_of_nodes(),nb_node_to_reach):
        graph.add_node(graph.number_of_nodes(), is_dummy=True)

def transform_3d_coordinates_into_ndarray(graph):
    """
    Transform the node attribute sphere_3dcoord from a list to a ndarray
    """
    # initialise the dict for atttributes on edges
    nodes_attributes = {}

    # Fill the dictionnary with the nd_array attribute
    for node in graph.nodes:
        nodes_attributes[node] = {"sphere_3dcoords": np.array(graph.nodes[node]["sphere_3dcoords"])}

    nx.set_node_attributes(graph, nodes_attributes)

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
    for graph_i, graph_file in enumerate([file_name for file_name in os.listdir(path_to_folder) if not os.path.isdir(os.path.join(path_to_folder, file_name))]):

        # load this graph
        graph = nx.read_gpickle(os.path.join(path_to_folder, graph_file))
        graph_list.append(graph)  # add it to the list of graph
        correspondence_dict[graph_i] = {"name":graph_file}  # add the information about the name.
        

    # transform the 3d attributes into ndarray
    for graph in graph_list:
        transform_3d_coordinates_into_ndarray(graph)
        
    # Calculate the geodesic distance for each node and add the id information
    for graph in graph_list:
        add_geodesic_distance_on_edges(graph)
        add_id_on_edges(graph)
        
    # add the number of nodes information to the dict and find the max number of nodes
    max_nb_nodes = 0
    for i, graph in enumerate(graph_list):
        correspondence_dict[i]["nb_nodes"] = graph.number_of_nodes()
        if graph.number_of_nodes() > max_nb_nodes:
            max_nb_nodes = graph.number_of_nodes()
   
    # add the dummy nodes
    for graph in graph_list:

        nx.set_node_attributes(graph, values=False, name="is_dummy")
        add_dummy_nodes(graph, max_nb_nodes)

    # Create the new folder for the graphs
    new_folder_path = os.path.join(path_to_folder, "modified_graphs")
    if not os.path.isdir(new_folder_path):
        os.mkdir(new_folder_path)

    # Save the graphs in the new folder
    for i, graph in enumerate(graph_list):
        graph_path = os.path.join(new_folder_path, "graph_"+str(i)+".gpickle")
        nx.write_gpickle(graph, graph_path)

    # Save the correspondence_dict
    pickle_out = open(os.path.join(path_to_folder,"correspondence_dict.pickle"),"wb")
    pickle.dump(correspondence_dict, pickle_out)
    pickle_out.close()



if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Given a set of graphs, create new graphs with equal sizes (by adding dummy nodes).")
    parser.add_argument("path_to_folder", help="path to a folder which contains the graphs")
    args = parser.parse_args()

    path_to_folder = args.path_to_folder

    read_modify_and_write_graphs(path_to_folder)

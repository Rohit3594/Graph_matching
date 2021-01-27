import os
import numpy as np
import networkx as nx


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
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    if True in is_dummy:
        graph_copy = graph.copy()
        graph_copy.remove_nodes_from(np.where(np.array(is_dummy) == True)[0])
        return graph_copy
    else:
        return graph

import numpy as np
import networkx as nx
import os
import argparse


def add_labels_to_graph(graph, list_labels):
    """
    Given a graph, add to each node the corresponding label found with dbscan
    """

    attribute_dict = {}
    for node in graph.nodes:
        attribute_dict[node] = {"label_dbscan":list_labels[node]}
    nx.set_node_attributes(graph, attribute_dict)

def read_and_modify_graphs(path_to_read, use_msync):
    """
    Read all graphs in the folder and add the label information to each node
    """

    # get the number of graphs
    nb_graphs = len(os.listdir(os.path.join(path_to_folder, "modified_graphs")))

    if use_msync == 0:
        # get the clustering labelling
        labelling_total = np.load(os.path.join(path_to_read,"X_clustering_labelling.npy"))
    elif use_msync == 1:
        labelling_total = np.load(os.path.join(path_to_read,"X_msync_labelling.npy"))
    else:
        print("not a correct value for the use_msync parameter, should be either 1 or 0")
    # get the labelling for each graph
    labelling_matrix = labelling_total.reshape(nb_graphs, -1)

    # Go through all the graphs
    for i_graph in range(nb_graphs):

        # load the graph
        path_to_graph = os.path.join(path_to_read, "modified_graphs","graph_"+str(i_graph)+".gpickle")
        graph = nx.read_gpickle(path_to_graph)

        # add the labelling information
        add_labels_to_graph(graph, labelling_matrix[i_graph,:])

        # save the graph
        nx.write_gpickle(graph, path_to_graph)


if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="add label information to the graphs with dbscan labelling information")
    parser.add_argument("path_to_folder", help="path to a folder which contains the graphs")
    parser.add_argument("--use_msync",
                        help="either to use dbscan labelling from mALS or use directly mSync results, 0 or 1",
                        default=0,
                        type=int)
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    use_msync = args.use_msync

    read_and_modify_graphs(path_to_folder, use_msync)
    

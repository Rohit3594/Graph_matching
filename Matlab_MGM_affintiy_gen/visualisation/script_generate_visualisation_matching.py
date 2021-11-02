''' This scripts aims at generating useful visualisation 
    for graph matching algorithms
    Author : Nathan Buskulic
    Creation date : March 20th 2020  
'''

import argparse
import numpy as np
import networkx as nx
import scipy.io as sio
import slam.mapping as smap
import trimesh
import matplotlib.pyplot as plt
import os


def get_planar_coords_in_graph(graph, distance_to_remove_outliers = 100):
    """
    Compute the planar coordinates for the graph
    and add them to each node in the graph
    """
    
    # Create the node array to be used in the mesh
    new_graph = graph.copy()
    node_array = np.array([new_graph.nodes[node]["coord"] for node in new_graph.nodes])
    # We also keep track of which node correspond to which line in the matrix
    node_correspondence = [node for i,node in enumerate(new_graph.nodes)]
    
    
    # transform in a mesh
    trimesh_test = trimesh.Trimesh(vertices=node_array)
    planar_coords = smap.stereo_projection(trimesh_test, h=-100)
    
    # Fix the coordinates of the north_pole and sphere radius
    radius = 100
    north_pole = np.array([0,0,100])
    
    # Fill the new coordinates
    dict_attributes = {}
    for i in range(new_graph.number_of_nodes()):
        
        node_num = node_correspondence[i]
        # Calculate the distance to north pole
        original_coords = new_graph.nodes[node_num]["coord"]
        distance_to_north_pole = np.linalg.norm(original_coords - north_pole)
        attr_val = np.array(planar_coords.vertices[i][:2])
        
        # If it is not an outlier :
        if np.linalg.norm(attr_val) < distance_to_remove_outliers:
            dict_attributes[node_num] = {"coord2D": attr_val}
        
        # otherwise we remove the node
        else:
            new_graph.remove_node(node_num)
        
    nx.set_node_attributes(new_graph,dict_attributes)
    return new_graph


def plot_and_save_superposition_of_graphs(ref_graph, noisy_graph, path_to_save_folder, acceptable_norm = 1000, nb_vertices=90, noise=0):
    """ 
    Get the 2D coordinates of the stereo projection
    and plot the corresponding graphs using networkx.
    """

    # Clean the figure
    plt.clf()

    # Get some useful data
    nb_outliers = len(ref_graph) - nb_vertices
    
    # Get the 2D coordinates
    noisy_graph = get_planar_coords_in_graph(noisy_graph,acceptable_norm)
    ref_graph = get_planar_coords_in_graph(ref_graph,acceptable_norm)

    # Prepare the position for the plot
    my_pos = {node: ref_graph.nodes[node]["coord2D"] for node in ref_graph.nodes}
    my_pos_noisy = {node: noisy_graph.nodes[node]["coord2D"] for node in noisy_graph.nodes}

    # Prepare the colors where outliers have different colors
    colors_ref = ["blue" if node < nb_vertices else "red" for node in ref_graph.nodes]
    colors_noisy = ["green" if node < nb_vertices else "m" for node in noisy_graph.nodes]

    # Prepare the matplotlib ax
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(13,9))
    # Draw the graphs
    nx.draw_networkx(ref_graph, 
                     pos=my_pos, 
                     with_labels=False, 
                     node_size=30, 
                     alpha=0.6, 
                     node_color=colors_ref,
                     edge_color="blue",
                     linewidths=0.5,
                     ax=ax, 
                     label="reference graph")

    nx.draw_networkx(noisy_graph,
                     pos=my_pos_noisy, 
                     with_labels=False, 
                     node_size=30, 
                     alpha=0.6, 
                     node_color=colors_noisy,
                     edge_color="green",
                     linewidths=0.5,
                     ax=ax, 
                     #style="dotted",
                     label="noisy graph")

    # Add blank lines to add elements to the legends
    ax.plot([], [], ' ', label=str(nb_vertices)+ " Vertices")
    ax.plot([], [], ' ', label=str(nb_outliers)+" Outliers")
    ax.plot([], [], ' ', label="Noise variance = "+str(noise))

    plt.title("Visualisation of a pair of graph with "+str(nb_outliers)+" outliers per graphs and a noise variance of "+str(noise))
    plt.legend()
    plt.savefig(os.path.join(path_to_save_folder, "superpos_graphs.svg"))
    plt.savefig(os.path.join(path_to_save_folder, "superpos_graphs.png"))

    
def plot_and_save_matching_result(ref_graph, noisy_graph, ground_truth, perm_matrix_algorithm, path_to_save_folder,
                                  algorithm_name="Algorithm", acceptable_norm=1000, nb_vertices = 90, noise=0):
    """
    Plot and save the 2D coordinates of the graphs where
    lines represents that two points are matched by the algorithm
    and where green line means that it is the right match,
    red line means that it is wrong, and black line means
    that it is related to an outlier.
    """

    # We create the permutation matrix of the ground truth
    g_t_permutation_matrix = np.zeros((len(ground_truth), len(ground_truth)))
    for i, elem in enumerate(ground_truth):
        g_t_permutation_matrix[i,elem] = 1

    # We create the graph that will hold the matching of the algorithm
    graph_matched = nx.Graph()

    # We add all the nodes from both graphs to the matched graph
    for node in ref_graph:
        graph_matched.add_node("r"+str(node), coord=ref_graph.nodes[node]["coord"])
    for node in noisy_graph:
        graph_matched.add_node("n"+str(node), coord=noisy_graph.nodes[node]["coord"])


    # We add the edges that corresponds to the matched nodes
    for ref_node in range(perm_matrix_algorithm.shape[0]):
        # We get the corresponding node in the noisy graph
        matched_node = np.where(perm_matrix_algorithm[ref_node,:] == 1)[0][0]

        # We see if it's a good match - A REVOIR
        if ref_node < g_t_permutation_matrix.shape[0] and matched_node < g_t_permutation_matrix.shape[1]:
            if g_t_permutation_matrix[ref_node,matched_node] == 1:
                color = "green"
            else:
                color="red"
        elif ref_node < g_t_permutation_matrix.shape[0] or matched_node < g_t_permutation_matrix.shape[1]:
            color = "red"
        else:
            color="black"
            
        # We add the edge that represent a match
        graph_matched.add_edge("r"+str(ref_node), "n"+str(matched_node), color=color)


    # Get the 2D coordinates
    graph_matched = get_planar_coords_in_graph(graph_matched,acceptable_norm)

    # get the edge colors
    color_edges = [graph_matched.edges[edge]["color"] for edge in graph_matched.edges]

    # Prepare the position for the plot
    pos = {node: graph_matched.nodes[node]["coord2D"] for node in graph_matched.nodes}

    # Prepare the colors where outliers have different colors
    colors_list = []
    for node in graph_matched.nodes:
        if int(node[1:]) < nb_vertices and node[0] == "r":
            colors_list.append("blue")
        if int(node[1:]) < nb_vertices and node[0] == "n":
            colors_list.append("green")
        if int(node[1:]) >= nb_vertices and node[0] == "r":
            colors_list.append("red")
        if int(node[1:]) >= nb_vertices and node[0] == "n":
            colors_list.append("m")

    # Prepare the matplotlib ax
    fig, ax = plt.subplots(nrows=1,ncols=1,figsize=(13,9))
    # Draw the graphs
    nx.draw_networkx(graph_matched, 
                     pos=pos, 
                     with_labels=False, 
                     node_size=30, 
                     alpha=0.6, 
                     node_color=colors_list,
                     edge_color=color_edges,
                     linewidths=0.5,
                     ax=ax, 
                     label="Matched points")

    # Add blank lines to add elements to the legends
    ax.plot([], [], ' ', label=str(nb_vertices)+ " Vertices")
    ax.plot([], [], ' ', label=str(len(ref_graph.nodes) - nb_vertices)+" Outliers")
    ax.plot([], [], ' ', label="Noise variance = "+str(noise))

    plt.title("Visualisation of a pair of graph with matching pairs of nodes given by "+algorithm_name+" algorithm")
    plt.legend()
    plt.savefig(os.path.join(path_to_save_folder, algorithm_name+"_matching.svg"))
    plt.savefig(os.path.join(path_to_save_folder, algorithm_name+"_matching.png"))





if __name__ == '__main__':

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="Generate visualisation of both graphs as well as the matching algorithms")
    parser.add_argument("path_to_read", help="path where the files for the graphs and the matchings are")
    parser.add_argument("path_to_write", help="path to a folder where to save the plots that will be generated")
    parser.add_argument("--noise", help="noise value that was used to generate the graphs", default=1, type=float)
    parser.add_argument("--nb_vertices", help="number of vertices that was used to generate the graphs", default=90, type=int)
    parser.add_argument("--acceptable_norm", help="The acceptable norm of a vector on the plane after stereo projection", default=1000, type=int)
    args = parser.parse_args()

    path_to_read = args.path_to_read
    path_to_write = args.path_to_write
    noise = args.noise
    nb_vertices = args.nb_vertices
    acceptable_norm = args.acceptable_norm


    # We get the noise value based on the file name
    try:
        pattern = "noise_"
        noise = float(path_to_read[path_to_read.find(pattern)+len(pattern):path_to_read.find(",")])
    except:
        print("Couldn't infer the noise value from the folder name")
    
    # Get the graphs and the ground truth matching
    ref_graph = nx.read_gpickle(os.path.join(path_to_read,"ref_graph.gpickle"))
    noisy_graph = nx.read_gpickle(os.path.join(path_to_read,"noisy_graph.gpickle"))
    ground_truth = np.load(os.path.join(path_to_read,"ground_truth.npy"))

    # plot the superposition of the two graphs
    plot_and_save_superposition_of_graphs(ref_graph, noisy_graph, path_to_write, noise=noise, acceptable_norm=acceptable_norm)

    perm_mat = np.transpose(sio.loadmat(os.path.join(path_to_read, "X_ipf.mat"))["X"])
    plot_and_save_matching_result(ref_graph, noisy_graph, ground_truth, perm_mat, path_to_write, "IPF", noise=noise, acceptable_norm=acceptable_norm)

    perm_mat = np.transpose(sio.loadmat(os.path.join(path_to_read, "X_rrwm.mat"))["X"])
    plot_and_save_matching_result(ref_graph, noisy_graph, ground_truth, perm_mat, path_to_write, "RRWM", noise=noise, acceptable_norm=acceptable_norm)

    perm_mat = np.transpose(sio.loadmat(os.path.join(path_to_read, "X_smac.mat"))["X"])
    plot_and_save_matching_result(ref_graph, noisy_graph, ground_truth, perm_mat, path_to_write, "SMAC", noise=noise, acceptable_norm=acceptable_norm)

    perm_mat = sio.loadmat(os.path.join(path_to_read, "X_kergm.mat"))["X"]
    plot_and_save_matching_result(ref_graph, noisy_graph, ground_truth, perm_mat, path_to_write, "KerGM", noise=noise, acceptable_norm=acceptable_norm)

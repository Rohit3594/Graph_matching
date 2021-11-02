import os
import numpy as np
import networkx as nx
import pickle

def load_graphs_in_list(path_to_graphs_folder):
    """
    Return a list of graph loaded from the path
    """
    list_graphs = []
    graph_files = []
    list_files = os.listdir(path_to_graphs_folder)
    for fil in list_files:
        if 'graph_' in fil:
            graph_files.append(fil)
    for graph_file in graph_files:
        path_graph = os.path.join(path_to_graphs_folder, graph_file)
        graph = nx.read_gpickle(path_graph)
        list_graphs.append(graph)

    return list_graphs


def graph_nodes_attribute(graph, attribute):
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def graph_edges_attribute(graph, attribute):
    att = list(nx.get_edge_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    if True in is_dummy:
        graph_copy = graph.copy()
        graph_copy.remove_nodes_from(np.where(np.array(is_dummy)==True)[0])
        return graph_copy
    else:
        return graph


def mean_degree(g):
    degree = g.degree
    mean_deg = 0
    for d in degree:
        mean_deg += d[1]
    return mean_deg/len(g)


def graph_metrics(real_graphs):
    # remove dummy nodes
    real_graphs_no_dummy = [remove_dummy_nodes(g) for g in real_graphs]

    # compare number of nodes
    nb_nodes_real_graphs = [len(g) for g in real_graphs_no_dummy]

    # compare nodes degree
    mean_degree_real_graphs = [mean_degree(g) for g in real_graphs_no_dummy]

    # compare 'geodesic_distance'
    geo_dist_real_graphs = [graph_edges_attribute(g, attribute='geodesic_distance') for g in real_graphs_no_dummy]

    mean_geo_dist_real_graphs = [np.mean(graph_edges) for graph_edges in geo_dist_real_graphs]

    std_geo_dist_real_graphs = [np.std(graph_edges) for graph_edges in geo_dist_real_graphs]

    return nb_nodes_real_graphs, mean_degree_real_graphs, mean_geo_dist_real_graphs, std_geo_dist_real_graphs



if __name__ == "__main__":

    path_to_real_graphs = '/mnt/data/work/python_sandBox/stage_nathan/data/OASIS_full_batch/modified_graphs'
    real_graphs = load_graphs_in_list(path_to_real_graphs)

    path_to_simu_graphs = '/mnt/data/work/python_sandBox/stage_nathan/data/simu_graphs/with_ref_for_visu'
    simus_spec = ['nb_vertices_10_noise_200_outliers_5/0', 'nb_vertices_10_noise_200_outliers_0/0','nb_vertices_10_noise_100_outliers_0/0','nb_vertices_85_noise_50_outliers_0/0','nb_vertices_85_noise_50_outliers_10/0','nb_vertices_85_noise_100_outliers_0/0' ]


    print('---------real data graphs----------')
    nb_nodes_real_graphs, mean_degree_real_graphs, mean_geo_dist_real_graphs, std_geo_dist_real_graphs = graph_metrics(real_graphs)
    print('mean nb of nodes', np.mean(nb_nodes_real_graphs))
    print('std nb of nodes', np.std(nb_nodes_real_graphs))
    print('mean degree', np.mean(mean_degree_real_graphs))
    print('mean mean geodesic distance', np.mean(mean_geo_dist_real_graphs))
    print('mean std geodesic distance', np.mean(std_geo_dist_real_graphs))

    for simu_spec in simus_spec:
        path_to_simu_graphs_spec = os.path.join(path_to_simu_graphs, simu_spec, 'graphs')
        print('---------simu graphs '+simu_spec+' ----------')
        simu_graphs = load_graphs_in_list(path_to_simu_graphs_spec)
        nb_nodes_simu_graphs, mean_degree_simu_graphs, mean_geo_dist_simu_graphs, std_geo_dist_simu_graphs = graph_metrics(simu_graphs)
        print('mean nb of nodes', np.mean(nb_nodes_simu_graphs))
        print('mean degree', np.mean(mean_degree_simu_graphs))
        print('mean mean geodesic distance', np.mean(mean_geo_dist_simu_graphs))
        print('mean std geodesic distance', np.mean(std_geo_dist_simu_graphs))


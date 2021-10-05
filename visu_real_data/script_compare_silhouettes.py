import os
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy
import matplotlib.pyplot as plt


def get_silhouette_per_cluster(silhouette_dict):
    nb_clusters = len(silhouette_dict)
    silhouette_data = np.zeros(nb_clusters)

    # Get the data
    for cluster_i, cluster_key in enumerate(silhouette_dict):
        silhouette_data[cluster_i] = np.mean(silhouette_dict[cluster_key])
    return silhouette_data


if __name__ == "__main__":
    # template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_full_batch/modified_graphs'
    path_to_silhouette = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_full_batch'
    methods = ['kerGM', 'mALS', 'mSync', 'hippi']

    nb_bins=10
    dens = False
    fig1, ax = plt.subplots(1, len(methods), sharey=True, sharex=True)

    clust_silhouettes = list()
    for ind, method in enumerate(methods):
        pickle_out = open(os.path.join(path_to_silhouette, 'labelling_'+method+'_silhouette.gpickle'), "rb")
        silhouette_dict = p.load(pickle_out)
        pickle_out.close()
        clust_silhouette = get_silhouette_per_cluster(silhouette_dict)
        clust_silhouettes.append(clust_silhouette)

        ax[ind].hist(clust_silhouette, density=dens, bins=nb_bins)  # density=False would make counts
        ax[ind].set_ylabel('Frequency')
        ax[ind].set_xlabel('Data')
        ax[ind].set_title(method)
        ax[ind].grid(True)

    plt.show()
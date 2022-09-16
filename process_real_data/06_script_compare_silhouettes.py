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
import tools.clusters_analysis as gca


if __name__ == "__main__":
    # template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/Oasis_original_new/'
    path_to_silhouette = '/mnt/data/work/python_sandBox/Graph_matching/data/Oasis_original_new_with_dummy/silhouette'
    methods = ['kerGM', 'mALS', 'mSync']

    nb_bins=10
    dens = False
    fig1, ax = plt.subplots(1, len(methods), sharey=True, sharex=True)

    clust_silhouettes = list()
    for ind, method in enumerate(methods):
        print('----------------------------')
        print(method)
        pickle_out = open(os.path.join(path_to_silhouette, 'labelling_'+method+'_silhouette.gpickle'), "rb")
        silhouette_dict = p.load(pickle_out)
        pickle_out.close()
        clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)
        nb_nodes = np.sum(clust_nb_nodes)
        clust_nb_nodes_perc = [100*v/nb_nodes for v in clust_nb_nodes]
        print(np.mean(clust_silhouette))
        print(np.std(clust_silhouette))
        print(len(clust_silhouette))
        print(silhouette_dict.keys())
        print(clust_nb_nodes_perc)
        if -0.1 in silhouette_dict.keys():
            ind_c=list(silhouette_dict.keys()).index(-0.1)
            print(clust_nb_nodes_perc[ind_c])
        clust_silhouettes.append(clust_silhouette)

        ax[ind].hist(clust_silhouette, density=dens, bins=nb_bins)  # density=False would make counts
        ax[ind].set_ylabel('Frequency')
        ax[ind].set_xlabel('Data')
        ax[ind].set_title(method)
        ax[ind].grid(True)

    plt.show()
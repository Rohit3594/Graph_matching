import sys
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import os
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy


if __name__ == "__main__":
    template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_labelled_pits_graphs'

    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    all_labels_neuroimage = list()
    all_labels_media = list()
    all_perc = list()
    for g in list_graphs:
        labels_media = list(nx.get_node_attributes(g, 'label_media').values())
        all_labels_media.extend(labels_media)
        a_labels_media = np.array(labels_media)
        perc_unlabelled = np.sum(a_labels_media==-2)/len(labels_media)
        all_perc.append(perc_unlabelled)
        labels_neuroimage = list(nx.get_node_attributes(g, 'label_neuroimage').values())
        all_labels_neuroimage.extend(labels_neuroimage)

    print(np.unique(all_labels_neuroimage))
    print(len(np.unique(all_labels_neuroimage)))
    print(np.unique(all_labels_media))
    print(len(np.unique(all_labels_media)))
    print(np.mean(np.array(all_perc)))
    print(all_perc)


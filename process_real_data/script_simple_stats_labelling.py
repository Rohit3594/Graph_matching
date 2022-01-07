import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
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


def unique_labels(all_labels):
    u_labs = list()
    for l in all_labels:
        u_labs.extend(l)
    return np.unique(u_labs)


def nb_labelled_nodes_per_label(u_labs, all_labels):
    u_l_count = list
    for u_l in u_labs:
        subj_u = list()
        for subj_labs in all_labels:
            subj_u.append(np.sum(subj_labs == u_l))
        u_l_count.append(subj_u)
    return u_l_count


if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/OASIS_labelled_pits_graphs'

    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    all_labels_neuroimage = list()
    all_labels_media = list()
    all_perc = list()
    nb_nodes = list()
    for g in list_graphs:
        nb_nodes.append(len(g.nodes()))
        labels_media = list(nx.get_node_attributes(g, 'label_media').values())
        all_labels_media.append(labels_media)
        a_labels_media = np.array(labels_media)
        perc_unlabelled = np.sum(a_labels_media==-2)/len(labels_media)
        all_perc.append(perc_unlabelled)
        labels_neuroimage = list(nx.get_node_attributes(g, 'label_neuroimage').values())
        all_labels_neuroimage.append(labels_neuroimage)

    print('average nb nodes:', np.mean(nb_nodes))
    print('std of nb nodes:', np.std(nb_nodes))

    u_neuroimage = unique_labels(all_labels_neuroimage)
    print('nb labels neuroimage:', len(u_neuroimage))
    print(u_neuroimage)

    u_media = unique_labels(all_labels_media)
    print('nb labels media:', len(u_media))
    print(u_media)

    print('average across individuals of the number of unlabelled nodes', np.mean(np.array(all_perc)))
    print(all_perc)


    nb_labelled_nodes_neuroimage = nb_labelled_nodes_per_label(u_neuroimage, all_labels_neuroimage)
    nb_labelled_nodes_media = nb_labelled_nodes_per_label(u_media, all_labels_media)

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
import matplotlib.pyplot as plt

if __name__ == "__main__":
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    mesh = sio.load_mesh(template_mesh)

    #path_to_graphs = '../data/OASIS_labelled_pits_graphs'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    method = 'neuroimage'#'media'#'kmeans_70_real_data_dummy'#'CAO'#'kerGM'#'mSync'#'mALS'#
    trash_label = -2#-0.1#-2
    reg_or_unreg = ''#'_unreg'#''
    largest_ind = 22#24

    path_to_X = "../data/Oasis_original_new_with_dummy/X_"+method+reg_or_unreg+".mat"

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for i,g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)

    if 'media' in method:
        label_attribute = 'label_media'
    elif 'neuroimage' in method:
        label_attribute = 'label_neuroimage'

    else:
        if ('kerGM' in method) or ('kmeans' in method):
            X = sco.loadmat(path_to_X)["full_assignment_mat"]
        else:
            X = sco.loadmat(path_to_X)['X']
        print(X.shape)
        print('get_clusters_from_assignment')
        # label_attribute = 'labelling_hippi'
        # gca.get_clusters_from_assignment_hippi(list_graphs, X_Hippi, largest_ind, mesh, label_attribute)
        label_attribute = 'labelling_' + method + reg_or_unreg
        gca.get_labelling_from_assignment(list_graphs, X, largest_ind, mesh, label_attribute)


    all_labels = list()
    all_perc = list()
    nb_nodes = list()
    nb_unlabelled = list()
    for g in list_graphs:
        nb_nodes.append(len(g.nodes()))
        labels_media = list(nx.get_node_attributes(g, label_attribute).values())
        all_labels.append(labels_media)
        a_labels = np.array(labels_media)
        perc_unlabelled = np.sum(a_labels==trash_label)/len(labels_media)
        nb_unlabelled.append(np.sum(a_labels==trash_label))
        all_perc.append(perc_unlabelled)

    print('average nb nodes:', np.mean(nb_nodes))
    print('std of nb nodes:', np.std(nb_nodes))

    u_labels = gca.unique_labels(all_labels)
    print('nb labels '+label_attribute+':', len(u_labels))
    print(u_labels)

    print('average across individuals of the number of unlabelled nodes', np.mean(np.array(all_perc)))
    print(all_perc)

    print('total percentage of unlabelled nodes', np.sum(nb_unlabelled)/np.sum(nb_nodes))

    nb_labelled_nodes = gca.nb_labelled_nodes_per_label(u_labels, all_labels)
    print(nb_labelled_nodes)
    print(nb_labelled_nodes.shape)
    nb_nodes_per_label = nb_labelled_nodes.sum(axis=1)
    print(nb_nodes_per_label/len(list_graphs))
    print(nb_labelled_nodes.max(axis=1))

    nb_bins=10
    dens = False
    fig1, ax = plt.subplots(1, 1, sharey=True, sharex=True)

    ax.hist(nb_nodes_per_label/len(list_graphs), density=dens, bins=nb_bins)  # density=False would make counts
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Data')
    ax.set_title(method)
    ax.grid(True)

    plt.show()
    #
    # #list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    # all_labels_neuroimage = list()
    # all_labels_media = list()
    # all_perc = list()
    # nb_nodes = list()
    # nb_unlabelled = list()
    # for g in list_graphs:
    #     nb_nodes.append(len(g.nodes()))
    #     labels_media = list(nx.get_node_attributes(g, 'label_media').values())
    #     all_labels_media.append(labels_media)
    #     a_labels_media = np.array(labels_media)
    #     perc_unlabelled = np.sum(a_labels_media==-2)/len(labels_media)
    #     nb_unlabelled.append(np.sum(a_labels_media==-2))
    #     all_perc.append(perc_unlabelled)
    #     labels_neuroimage = list(nx.get_node_attributes(g, 'label_neuroimage').values())
    #     all_labels_neuroimage.append(labels_neuroimage)
    #
    # print('average nb nodes:', np.mean(nb_nodes))
    # print('std of nb nodes:', np.std(nb_nodes))
    #
    # u_neuroimage = gca.unique_labels(all_labels_neuroimage)
    # print('nb labels neuroimage:', len(u_neuroimage))
    # print(u_neuroimage)
    #
    # u_media = gca.unique_labels(all_labels_media)
    # print('nb labels media:', len(u_media))
    # print(u_media)
    #
    # print('average across individuals of the number of unlabelled nodes', np.mean(np.array(all_perc)))
    # print(all_perc)
    #
    # print('total percentage of unlabelled nodes', np.sum(nb_unlabelled)/np.sum(nb_nodes))
    #
    #
    #
    # nb_labelled_nodes_media = gca.nb_labelled_nodes_per_label(u_media, all_labels_media)
    # print('media')
    # print(nb_labelled_nodes_media)
    # print(nb_labelled_nodes_media.shape)
    # print(nb_labelled_nodes_media.sum(axis=1))
    # nb_labelled_nodes_neuroimage = gca.nb_labelled_nodes_per_label(u_neuroimage, all_labels_neuroimage)
    # print('neuroimage')
    # print(nb_labelled_nodes_neuroimage)
    # print(nb_labelled_nodes_neuroimage.shape)
    # print(nb_labelled_nodes_neuroimage.sum(axis=1))
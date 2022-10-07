import os
import sys
#sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy
from visbrain.objects import SourceObj, ColorbarObj


if __name__ == "__main__":
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_silhouette = '../data/Oasis_original_new_with_dummy/silhouette'
    path_to_figs = '../data/Oasis_original_new_with_dummy/figures'

    reg_or_unreg = ''#'_unreg'#''
    method = 'media'#'kmeans_70_real_data_dummy'#'CAO'#'kerGM'#'mSync'#'mALS'#

    path_to_X = "../data/Oasis_original_new_with_dummy/X_"+method+reg_or_unreg+".mat"
    template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'#lh.OASIS_testGrp_average_inflated.gii'
    largest_ind = 22#24

    mesh = sio.load_mesh(template_mesh)
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for i,g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        #print(i,len(g)) # allow to identify the largest graph (22)

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
        #label_attribute = 'labelling_hippi'
        #gca.get_clusters_from_assignment_hippi(list_graphs, X_Hippi, largest_ind, mesh, label_attribute)
        label_attribute = 'labelling_'+method+reg_or_unreg
        gca.get_labelling_from_assignment(list_graphs, X, largest_ind, mesh, label_attribute)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)

    print('load precomputed silhouette value')
    pickle_in = open(os.path.join(path_to_silhouette, 'labelling_'+method+reg_or_unreg+'_silhouette.gpickle'), "rb")
    silhouette_dict = p.load(pickle_in)
    pickle_in.close()

    clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)

    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                        nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
                                                        vmin=-1, vmax=1)

    vb_sc.add_to_subplot(s_obj)
    # vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1,
    #                            col=3, width_max=200)
    vb_sc.preview()
    # Ry(180)
    transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    reg_mesh.apply_transform(transfo_full)
    vb_sc1 = gv.visbrain_plot(reg_mesh)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=clust_silhouette,
                                                        nodes_size=60, nodes_mask=None, c_map='jet', symbol='disc',
                                                        vmin=-1, vmax=1)

    vb_sc1.add_to_subplot(s_obj)
    # vb_sc.add_to_subplot(nodes_cb_obj, row=visb_sc_shape[0] - 1,
    #                            col=3, width_max=200)
    vb_sc1.preview()

    #vb_sc.screenshot(os.path.join(path_to_figs, 'silhouette_'+label_attribute+'.png'))
    print(np.mean(clust_silhouette))
    print(np.std(clust_silhouette))
    print(len(clust_silhouette))
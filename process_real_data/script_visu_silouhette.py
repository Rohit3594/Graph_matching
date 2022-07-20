import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
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
    # template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/OASIS_full_batch/modified_graphs'
    path_to_silhouette = '../data/OASIS_full_batch'
    path_to_mALS = "../data/OASIS_full_batch/X_mALS.mat"
    path_to_mSync = "../data/OASIS_full_batch/X_mSync.mat"
    path_to_CAO = "../data/OASIS_full_batch/X_cao_cst_o.mat"
    path_to_kerGM = "../data/OASIS_full_batch/X_pairwise_kergm.mat"
    # path_to_match_mat = "/home/rohit/PhD_Work/GM_my_version/RESULT_FRIOUL_HIPPI/Hippi_res_real_mat.npy"

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    X_mALS = sco.loadmat(path_to_mALS)['X']
    X_mSync = sco.loadmat(path_to_mSync)['X']
    X_CAO = sco.loadmat(path_to_CAO)['X']
    X_kerGM = sco.loadmat(path_to_kerGM)["full_assignment_mat"]

    # path_to_match_mat = "../data/OASIS_full_batch/Hippi_res_real_mat.npy"
    # X_Hippi = np.load(path_to_match_mat)
    #label_attribute = 'labelling_CAO' # must be changed also

    label_attribute = 'labelling_kerGM'

    mesh = sio.load_mesh(template_mesh)
    largest_ind=24
    print('get_clusters_from_assignment')
    #gca.get_clusters_from_assignment_hippi(list_graphs, X_Hippi, largest_ind, mesh, label_attribute)
    #gca.get_clusters_from_assignment(list_graphs, X_CAO, largest_ind, mesh, label_attribute)

    gca.get_clusters_from_assignment(list_graphs, X_kerGM, largest_ind, mesh, label_attribute)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict)

    # Calculate or load the silhouette values
    # if path_silhouette != "":
    #     pickle_in = open(path_silhouette, "rb")
    #     silhouette_dict = pickle.load(pickle_in)
    #     pickle_in.close()
    #
    # else:
    print('get_all_silhouette_value')
    # silhouette_dict = gca.get_all_silhouette_value(list_graphs, cluster_dict)
    # pickle_out = open(os.path.join(path_to_silhouette, label_attribute+'_silhouette.gpickle'), "wb")
    # p.dump(silhouette_dict, pickle_out)
    pickle_out = open(os.path.join(path_to_silhouette, label_attribute+'_silhouette.gpickle'), "rb")
    silhouette_dict = p.load(pickle_out)
    pickle_out.close()
    clust_silhouette, clust_nb_nodes = gca.get_silhouette_per_cluster(silhouette_dict)

    # # save the silhouette value if necessary
    # if path_to_save != "":
    #     pickle_out = open(path_to_save, "wb")
    #     pickle.dump(silhouette_dict, pickle_out)
    #     pickle_out.close()
    print('visu')
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)
    s_obj, cb_obj = gca.get_silhouette_source_obj(centroid_dict,
                                              list_graphs,
                                              clust_silhouette,
                                              mesh, c_map='jet', clim=(-1,1))

    vb_sc.add_to_subplot(s_obj)
    # visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=3, width_max=200)
    vb_sc.preview()
    #vb_sc.screenshot(os.path.join(path_to_silhouette, label_attribute+'.png'))
    print(np.mean(clust_silhouette))
    print(np.std(clust_silhouette))
    print(len(clust_silhouette))
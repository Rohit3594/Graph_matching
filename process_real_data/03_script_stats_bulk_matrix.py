import sys

#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import matplotlib.pyplot as plt
import os
from visbrain.objects import SourceObj, ColorbarObj

if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    c_map = 'hot'
    vmin = 80
    vmax = 100
    #path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_full_batch/modified_graphs'
    #path_to_match_mat = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_full_batch'
    #x_mSync = sco.loadmat(os.path.join(path_to_match_mat, "X_mSync.mat"))["X"]
    #x_mALS = sco.loadmat(os.path.join(path_to_match_mat, "X_mALS.mat"))["X"]
    #x_cao = sco.loadmat(os.path.join(path_to_match_mat, "X_cao_cst_o.mat"))["X"]
    #Hippi = np.load(os.path.join(path_to_match_mat, "Hippi_res_real_mat.npy"))
    #x_kerGM = sco.loadmat(os.path.join(path_to_match_mat,"X_pairwise_kergm.mat"))["full_assignment_mat"]
    #nb_graphs = 134

    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_match_mat = '../data/Oasis_original_new_with_dummy/'

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    nb_graphs = len(list_graphs)
    # g=list_graphs[0]
    # mask_r = list(nx.get_node_attributes(g, "is_dummy").values())
    # print(np.sum(np.logical_not(mask_r)))
    # data_mask = gp.remove_dummy_nodes(g)
    # print(np.sum(data_mask))
    # print(data_mask.shape)
    # print(len(mask_r))

    # compute the mask of dummy nodes
    is_dummy_vect = []
    for g in list_graphs:
        is_dummy_vect.extend(list(nx.get_node_attributes(g, "is_dummy").values()))
    not_dummy_vect = np.logical_not(is_dummy_vect)
    print(len(is_dummy_vect))#_vect))
    print(len(not_dummy_vect))
    print(np.sum(is_dummy_vect))
    print(np.sum(not_dummy_vect))

    # read the assignment matrices
    x_mSync = sco.loadmat(os.path.join(path_to_match_mat, "X_mSync.mat"))["X"]
    x_mALS = sco.loadmat(os.path.join(path_to_match_mat, "X_mALS.mat"))["X"]
    x_cao = sco.loadmat(os.path.join(path_to_match_mat, "X_CAO.mat"))["X"]
    x_kerGM = sco.loadmat(os.path.join(path_to_match_mat,"X_kerGM.mat"))["full_assignment_mat"]

    # compute for each row of the assignment matrix the percent of matched nodes across the graphs
    match_no_dummy_mSync = 100*np.sum(x_mSync[:, not_dummy_vect],1)/ nb_graphs
    match_dummy_mSync = 100*np.sum(x_mSync[:, is_dummy_vect],1)/ nb_graphs
    match_no_dummy_mALS = 100*np.sum(x_mALS[:, not_dummy_vect],1)/ nb_graphs
    match_dummy_mALS = 100*np.sum(x_mALS[:, is_dummy_vect],1)/ nb_graphs
    match_no_dummy_kerGM = 100*np.sum(x_kerGM[:, not_dummy_vect],1)/ nb_graphs
    match_dummy_kerGM = 100*np.sum(x_kerGM[:, is_dummy_vect],1)/ nb_graphs

    #match_no_dummy_Hippi = np.sum(Hippi,1)/ nb_graphs
    #match_no_dummy_mSync = 100*np.sum(x_mSync[:, not_dummy_vect],1)/ nb_graphs
    #match_dummy_mSync = 100*np.sum(x_mSync[:, is_dummy_vect],1)/ nb_graphs

    # plot the ditribution across the rows of the matrix
    nb_bins=50
    dens = False
    # fig1, ax = plt.subplots(1,2)
    # ax[1].hist(match_dummy_mSync, density=True, bins=nb_bins)  # density=False would make counts
    # ax[1].set_ylabel('Frequency')
    # ax[1].set_xlabel('Data')
    # ax[1].set_title('dummy match for mSync')
    # fig2, ax = plt.subplots(1,2)
    # ax[1].hist(match_dummy_mALS, density=True, bins=nb_bins)  # density=False would make counts
    # ax[1].set_ylabel('Frequency')
    # ax[1].set_xlabel('Data')
    # ax[1].set_title(' dummy match for mALS')
    fig1, ax = plt.subplots(1, 3, sharey=True)
    ax[0].hist(match_no_dummy_kerGM, density=dens, bins=nb_bins)  # density=False would make counts
    ax[0].set_ylabel('Frequency')
    ax[0].set_xlabel('Data')
    ax[0].set_title('no dummy match for kerGM')
    # ax[1].hist(match_no_dummy_Hippi, density=dens, bins=nb_bins)  # density=False would make counts
    # ax[1].set_ylabel('Frequency')
    # ax[1].set_xlabel('Data')
    # ax[1].set_title('no dummy match for Hippi')
    ax[1].hist(match_no_dummy_mALS, density=dens, bins=nb_bins)  # density=False would make counts
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlabel('Data')
    ax[1].set_title('no dummy match formALS')
    ax[2].hist(match_no_dummy_mSync, density=dens, bins=nb_bins)  # density=False would make counts
    ax[2].set_ylabel('Frequency')
    ax[2].set_xlabel('Data')
    ax[2].set_title('no dummy match for mSync')
    plt.show()

    # visu on the mesh
    # # Get the mesh
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))
    vb_sc = gv.visbrain_plot(mesh)
    vb_sc = gv.visbrain_plot(mesh, caption='mSync')
    vb_sc2 = gv.visbrain_plot(mesh, caption='mALS')
    vb_sc3 = gv.visbrain_plot(mesh, caption='mKerGM')
    #vb_sc4 = gv.visbrain_plot(mesh, caption='mHippi')

    for i in range(nb_graphs):
        g=list_graphs[i]
        #match_label_per_graph = {}
        nb_nodes = len(g.nodes)
        scope = range(i * nb_nodes, (i + 1) * nb_nodes)
        data_match_dummy_mSync = match_dummy_mSync[scope]
        data_match_no_dummy_mSync = match_no_dummy_mSync[scope]
        data_match_dummy_mALS = match_dummy_mALS[scope]
        data_match_no_dummy_mALS = match_no_dummy_mALS[scope]
        data_match_dummy_kerGM = match_dummy_kerGM[scope]
        data_match_no_dummy_kerGM = match_no_dummy_kerGM[scope]

        data_mask = gp.remove_dummy_nodes(g)

        print(np.min(data_match_no_dummy_mSync[data_mask]))
        print(np.max(data_match_no_dummy_mSync[data_mask]))

        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
        # s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_mSync[data_mask], clim=clim)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_match_no_dummy_mSync[data_mask],
                                                        nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)
        vb_sc.add_to_subplot(s_obj)

        #s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_mALS[data_mask], clim=clim)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_match_no_dummy_mALS[data_mask],
                                                        nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)
        vb_sc2.add_to_subplot(s_obj)

        #s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_kerGM[data_mask], clim=clim)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_match_no_dummy_kerGM[data_mask],
                                                        nodes_mask=None, c_map=c_map, symbol='disc',
                                                        vmin=vmin, vmax=vmax)
        vb_sc3.add_to_subplot(s_obj)

    # visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=3, width_max=200)

        # vb_sc2 = gv.visbrain_plot(mesh)
        # s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_dummy_mSync[data_mask], clim=clim)
        # visb_sc_shape = gv.get_visb_sc_shape(vb_sc2)
        # vb_sc2.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1] - 1)
        # vb_sc2.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
        #                      col=visb_sc_shape[1] + 1, width_max=200)
        # vb_sc2.preview()
    # curr_node=0
    # for i in range(nb_graphs):
    #     g = list_graphs[i]
    #     gp.remove_dummy_nodes(g)
    #     #match_label_per_graph = {}
    #     nb_nodes = len(g.nodes)
    #     print(nb_nodes)
    #     scope = range(curr_node, curr_node+nb_nodes)
    #     curr_node = curr_node+nb_nodes
    #
    #     data_match_no_dummy_Hippi = match_no_dummy_Hippi[scope]
    #     s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_match_no_dummy_Hippi, clim=clim)
    #     vb_sc4.add_to_subplot(s_obj)
    #
    # visb_sc_shape = gv.get_visb_sc_shape(vb_sc3)
    # vb_sc3.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=200)
    vb_sc.preview()
    vb_sc2.preview()
    vb_sc3.preview()
    #vb_sc4.preview()



    # list_graphs = gp.load_graphs_in_list(path_to_graphs)
    # for g in list_graphs:
    #     gp.remove_dummy_nodes(g)
    #     print(len(g))

    # # Get the mesh
    # mesh = sio.load_mesh(template_mesh)
    # vb_sc = gv.visbrain_plot(mesh)
    # # gp.remove_dummy_nodes(g)
    # # label_nodes_according_to_coord(g, mesh, coord_dim=1)
    # # nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    # # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    # # vb_sc.add_to_subplot(s_obj)
    # # vb_sc.preview()

    # for ind_g, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)
    #     label_nodes_according_to_coord(g, mesh, coord_dim=1)
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    #     vb_sc.add_to_subplot(s_obj)

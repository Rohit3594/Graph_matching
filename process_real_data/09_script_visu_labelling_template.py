import sys
import os
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p


if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'#'../data/template_mesh/OASIS_avg.lh.white.talairach.unreg.ico7.gii'#'../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'

    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_match_mat = '../data/Oasis_original_new_with_dummy/'

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    nb_graphs = len(list_graphs)
    tot_nb_nodes = 0
    for i, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        tot_nb_nodes += len(g)

    # read the assignment matrices
    x_mSync = sco.loadmat(os.path.join(path_to_match_mat, "X_mSync.mat"))["X"]
    x_mALS = sco.loadmat(os.path.join(path_to_match_mat, "X_mALS_unreg.mat"))["X"]
    x_cao = sco.loadmat(os.path.join(path_to_match_mat, "X_cao_cst_o.mat"))["X"]
    x_kerGM = sco.loadmat(os.path.join(path_to_match_mat,"X_pairwise_kergm.mat"))["full_assignment_mat"]

    X = x_mALS

    mesh = sio.load_mesh(template_mesh)
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)
    largest_ind=22
    label_attribute = 'labelling_from_assgn'

    gca.get_labelling_from_assignment(list_graphs, X, largest_ind, reg_mesh, label_attribute)

    print('create_clusters_lists')
    cluster_dict = gca.create_clusters_lists(list_graphs, label_attribute=label_attribute)
    # Calculate the centroid
    print('get_centroid_clusters')
    centroid_dict = gca.get_centroid_clusters(list_graphs, cluster_dict, coords_attribute="sphere_3dcoords_noreg")

    vb_sc = gv.visbrain_plot(reg_mesh)
    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index_noreg', reg_mesh)
        #labels = nx.get_node_attributes(g, 'label_media').values()
        labels = nx.get_node_attributes(g, label_attribute).values()
        color_label = np.array([l for l in labels])
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None,
                                                        c_map='nipy_spectral',  vmin=-0.1, vmax=1)
        vb_sc.add_to_subplot(s_obj)

    centroids_3Dpos = gca.get_centroids_coords(centroid_dict, list_graphs, reg_mesh, attribute_vertex_index='ico100_7_vertex_index_noreg')
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(centroids_3Dpos, node_data=np.array(list(cluster_dict.keys())),
                                                        nodes_size=60, nodes_mask=None, c_map='nipy_spectral', symbol='disc',
                                                        vmin=-0.1, vmax=1)

    vb_sc.add_to_subplot(s_obj)
    vb_sc.preview()

    nb_unmatched_nodes = 0
    if -0.1 in cluster_dict.keys():
        nb_unmatched_nodes = len(cluster_dict[-0.1])
    print('percent nb_unmatched_nodes=', 100*nb_unmatched_nodes/tot_nb_nodes)



    # default_value = -0.1#0.05
    # nb_nodes = len(g_l.nodes)
    #
    #
    #
    # for j in range(len(list_graphs)):
    #
    #     grph = list_graphs[j]
    #     nodes_to_remove = gp.remove_dummy_nodes(grph)
    #     nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
    #
    #     grph.remove_nodes_from(list(nodes_to_remove[0]))
    #     nb_nodes = len(grph.nodes)
    #     row_scope = range(j * nb_nodes, (j + 1) * nb_nodes)
    #
    #     print(len(grph.nodes))
    #
    #     if len(grph.nodes)==101:
    #         break
    #
    #
    # for matching_matrix in X:
    #
    #     print(matching_matrix.shape)
    #     last_index = 0
    #
    #     nb_unmatched = 0
    #     for i in range(nb_graphs):
    #
    #         #g = list_graphs[i]
    #         g=p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
    #
    #         nodes_to_remove = gp.remove_dummy_nodes(g)
    #         nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
    #         g.remove_nodes_from(list(nodes_to_remove[0]))
    #         nb_nodes = len(g.nodes)
    #
    #         print(len(g.nodes))
    #
    #         if i == 0:
    #             col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #             prev_nb_nodes = nb_nodes
    #             perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
    #             transfered_labels = np.ones(nb_nodes)*default_value
    #             last_index+=nb_nodes
    #         else:
    #             col_scope = range(last_index, last_index+nb_nodes)
    #             last_index += nb_nodes
    #             perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
    #             transfered_labels = np.ones(nb_nodes)*default_value
    #
    #
    #         print(col_scope)
    #
    #         #nb_nodes = len(g.nodes)
    #         #col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #
    #         for node_indx,ind in enumerate(row_scope):
    #             match_index = np.where(perm_X[node_indx,:]==1)[0]
    #
    #             if len(match_index)>0:
    #                 transfered_labels[match_index[0]] = color_label[node_indx]
    #         nb_unmatched += np.sum(transfered_labels==default_value)
    #         #data_mask = gp.remove_dummy_nodes(g)
    #         nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
    #         print(nodes_coords.shape)
    #         s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=transfered_labels, nodes_mask=None, c_map='nipy_spectral', symbol='disc', vmin=0, vmax=1)
    #
    #
    #         vb_sc.add_to_subplot(s_obj)
    #     print('nb_unmatched',nb_unmatched)
    #     print("Preview")
    #     vb_sc.preview()



        # for l in range(len(g_l)):
        #     index = np.where()
        #     transfered_labels[index]=color_label[l]




    # is_dummy = []
    # for i in range(nb_graphs):
    #     sing_graph = p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
    #     is_dummy.append(list(nx.get_node_attributes(sing_graph,"is_dummy").values()))
    #
    # is_dummy_vect = [val for sublist in is_dummy for val in sublist]
    #
    # # # Get the mesh
    # mesh = sio.load_mesh(template_mesh)
    # vb_sc = gv.visbrain_plot(mesh)
    #
    # for i in range(nb_graphs):
    #     match_label_per_graph={}
    #
    #     g = p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
    #     nb_nodes = len(g.nodes)
    #     scope = range(i * nb_nodes, (i + 1) * nb_nodes)
    #     for node_indx,ind in enumerate(scope):
    #         match_indexes = np.where(matching_matrix[ind,:]==1)[0]
    #         match_perc = (len(match_indexes) - len(set(match_indexes).intersection(np.where(np.array(is_dummy_vect)==True)[0])))/nb_graphs
    #         match_label_per_graph[node_indx] = {'label_color':match_perc}
    #
    #     nx.set_node_attributes(g, match_label_per_graph)
    #
    #     gp.remove_dummy_nodes(g)
    #
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     node_data = gp.graph_nodes_attribute(g, "label_color")
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_data=node_data, nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
    #


    # list_graphs = gp.load_graphs_in_list(path_to_graphs)
    # for g in list_graphs:
    #     gp.remove_dummy_nodes(g)
    #     print(len(g))

    # # Get the mesh
    # mesh = sio.load_mesh(template_mesh)
    # vb_sc = gv.visbrain_plot(mesh)
    # gp.remove_dummy_nodes(g)
    # label_nodes_according_to_coord(g, mesh, coord_dim=1)
    # nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    # vb_sc.add_to_subplot(s_obj)
    # vb_sc.preview()

    # for ind_g, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)
    #     label_nodes_according_to_coord(g, mesh, coord_dim=1)
    #     nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    #     node_data = gp.graph_nodes_attribute(g, "label_color")
    #     s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_data=node_data, nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    #     vb_sc.add_to_subplot(s_obj)

    # vb_sc.preview()
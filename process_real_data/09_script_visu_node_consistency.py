import os
import slam.io as sio
import numpy as np
import networkx as nx
import pickle
from visbrain.objects import SourceObj, ColorbarObj
import tools.graph_visu as gv
import tools.graph_processing as gp

if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    path_to_figs = '../data/Oasis_original_new_with_dummy/figures'
    reg_or_unreg = ''#'_unreg'#''
    ind_g=22
    method = 'kmeans_110_real_data_dummy'#'mALS'#'kerGM'#'mSync'#'mALS'+reg_or_unreg

    if reg_or_unreg == '_unreg':
        graph_att = '_noreg'
    else:
        graph_att = ''

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    print(len(list_graphs))

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_"+method+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph= pickle.load(pickle_in)
    pickle_in.close()


    # pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_Hippi.pck"),"rb")
    # nodeCstPerGraph_Hippi = pickle.load(pickle_in)
    # pickle_in.close()


    print("Node consistency :", np.mean(nodeCstPerGraph), np.std(nodeCstPerGraph))

    print(np.mean(nodeCstPerGraph, 1))
    print(np.std(nodeCstPerGraph, 1))


    vmin = 0.7
    vmax = 1

    # for g in list_graphs:
    #     #gp.remove_dummy_nodes(g)
    #     print(len(g))
    # Get the mesh
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    vb_sc = gv.visbrain_plot(mesh)#None
    for ind_g, g in enumerate(list_graphs):
        data_mask = gp.remove_dummy_nodes(g)
        data_node_cstr = nodeCstPerGraph[:,ind_g]
        #vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index'+graph_att, mesh)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
                                                        nodes_size=None, nodes_mask=None, c_map='hot', symbol='disc',
                                                        vmin=vmin, vmax=vmax)



        vb_sc.add_to_subplot(s_obj)
        #visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
        #vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    #visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    #vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=200)
    vb_sc.preview()
    #vb_sc.screenshot(os.path.join(path_to_figs, 'consistency_'+method+'.png'))

#
#     #
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    vb_sc1 = gv.visbrain_plot(mesh)
    #clim=(0.8, 0.95)
    g=list_graphs[ind_g]
    data_mask = gp.remove_dummy_nodes(g)
    data_node_cstr = np.mean(nodeCstPerGraph,1)
    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index'+graph_att, mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
                                                        nodes_size=60, nodes_mask=None, c_map='hot', symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc1.add_to_subplot(s_obj)
    vb_sc1.preview()

    #vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=60)
    # Ry(180)
    transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index'+graph_att, mesh)
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=data_node_cstr[data_mask],
                                                        nodes_size=60, nodes_mask=None, c_map='hot', symbol='disc',
                                                        vmin=vmin, vmax=vmax)

    vb_sc2 = gv.visbrain_plot(mesh)
    vb_sc2.add_to_subplot(s_obj)
    vb_sc2.preview()


    # vb_sc = None
    # clim=(0.8, 0.95)
    # s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_Hippi,1), clim=clim, transl=[0,0,2])
    # vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    # visb_sc_shape = get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    #
    # # Ry(180)
    # transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    # mesh.apply_transform(transfo_full)
    # s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_Hippi,1), clim=clim, transl=[0,0,2])
    # vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    # visb_sc_shape = get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                            col=visb_sc_shape[1] + 1, width_max=200)
    # vb_sc.preview()
    #
    # print(np.min(np.mean(nodeCstPerGraph_Hippi, 1)))
    # print(np.max(np.mean(nodeCstPerGraph_Hippi, 1)))

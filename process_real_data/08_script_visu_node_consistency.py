import os
import slam.io as sio
import numpy as np
import networkx as nx
import pickle
from visbrain.objects import SourceObj, ColorbarObj
import tools.graph_visu as gv
import tools.graph_processing as gp


def show_graph_nodes(graph, mesh, data, clim=(0, 1), transl=None):

    # manage nodes
    s_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', mesh)
    print("s_coords",s_coords.shape)

    transl_bary = np.mean(s_coords)
    s_coords = 1.01*(s_coords-transl_bary)+transl_bary

    if transl is not None:
        s_coords += transl

    s_obj = SourceObj('nodes', s_coords, color='red',#data=data[data_mask],
                        edge_color='black', symbol='disc', edge_width=2.,
                        radius_min=30., radius_max=30., alpha=.9)
    """Color the sources according to data
    """
    s_obj.color_sources(data=data, cmap='hot', clim=clim)
    # Get the colorbar of the source object
    CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), txtcolor='k')
    cb_obj = ColorbarObj(s_obj, cblabel='node consistency', border=False,
                  **CBAR_STATE)

    return s_obj, cb_obj


if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_mALS.pck"),"rb")
    nodeCstPerGraph_mALS = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_mSync.pck"),"rb")
    nodeCstPerGraph_mSync = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_CAO.pck"),"rb")
    nodeCstPerGraph_CAO = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_KerGM.pck"),"rb")
    nodeCstPerGraph_KerGM = pickle.load(pickle_in)
    pickle_in.close()

    # pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_Hippi.pck"),"rb")
    # nodeCstPerGraph_Hippi = pickle.load(pickle_in)
    # pickle_in.close()


    print("Node consistency mALS:",np.mean(nodeCstPerGraph_mALS), np.std(nodeCstPerGraph_mALS))
    print("Node consistency mSync:",np.mean(nodeCstPerGraph_mSync), np.std(nodeCstPerGraph_mSync))
    print("Node consistency KerGM:",np.mean(nodeCstPerGraph_KerGM), np.std(nodeCstPerGraph_KerGM))
    print("Node consistency CAO:",np.mean(nodeCstPerGraph_CAO))
    # print("Node consistency Hippi:",np.mean(nodeCstPerGraph_Hippi), np.std(nodeCstPerGraph_Hippi))

    print(np.mean(nodeCstPerGraph_mALS,1))
    print(np.std(nodeCstPerGraph_mALS,1))
    #print(np.mean(nodeCstPerGraph_mSync,1))
    #print(np.mean(nodeCstPerGraph_KerGM,1))
    #print(np.mean(nodeCstPerGraph_CAO,1))
    #rank_mSync = np.linalg.matrix_rank(matching_mSync)
    #print(rank_mSync)

    # for g in list_graphs:
    #     #gp.remove_dummy_nodes(g)
    #     print(len(g))
    # Get the mesh
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    vb_sc = gv.visbrain_plot(mesh)#None
    clim=(0.7, 1)
    for ind_g, g in enumerate(list_graphs):
        data_mask = gp.remove_dummy_nodes(g)
        data_node_cstr = nodeCstPerGraph_mALS[:,ind_g]
        #vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
        s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_node_cstr[data_mask], clim=clim)
        vb_sc.add_to_subplot(s_obj)
        #visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
        #vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    #visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
    #vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=200)
    vb_sc.preview()
#
#     #
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    vb_sc1 = gv.visbrain_plot(mesh)
    #clim=(0.8, 0.95)
    ind_g=19
    g=list_graphs[ind_g]
    data_mask = gp.remove_dummy_nodes(g)
    data_node_cstr = np.mean(nodeCstPerGraph_mALS,1)
    nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_node_cstr[data_mask], clim=clim)
    visb_sc_shape = gv.get_visb_sc_shape(vb_sc1)
    vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    #vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                           col=visb_sc_shape[1] + 1, width_max=60)
#     # Ry(180)
#     transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
#     mesh.apply_transform(transfo_full)
#     s_obj, cb_obj = show_graph_nodes(g, mesh, data=data_node_cstr[data_mask], clim=clim)
#     vb_sc1 = gv.visbrain_plot(mesh, visb_sc=vb_sc1)
#     visb_sc_shape = gv.get_visb_sc_shape(vb_sc1)
#     vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
#     vb_sc1.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
#                                col=visb_sc_shape[1] + 1, width_max=200)
#
    vb_sc1.preview()
    print(np.min(np.mean(nodeCstPerGraph_mALS, 1)))
    print(np.max(np.mean(nodeCstPerGraph_mALS, 1)))

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

import os
import slam.io as sio
import numpy as np
import networkx as nx
import pickle
from visbrain.objects import SourceObj, ColorbarObj


def get_visb_sc_shape(visb_sc):
    """
    get the subplot shape in a visbrain scene
    :param visb_sc:
    :return: tuple (number of rows, number of coloumns)
    """
    k = list(visb_sc._grid_desc.keys())
    return k[-1]


def graph_nodes_to_coords(graph, index_attribute, mesh):
    vert_indices = list(nx.get_node_attributes(graph, index_attribute).values())
    coords = np.array(mesh.vertices[vert_indices, :])
    return coords[:88]


def graph_nodes_attribute(graph, attribute):
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    print(is_dummy)
    print(len(graph))
    print("is_dummy:",is_dummy.shape)
    data_mask = np.ones_like(is_dummy)
    if True in is_dummy:
        graph_copy = graph.copy()
        inds_dummy = np.where(np.array(is_dummy)==True)[0]
        graph_copy.remove_nodes_from(inds_dummy)
        data_mask[inds_dummy] = 0
        print(data_mask)
        print("number of nodes:",len(graph_copy.nodes()))
        return graph_copy, data_mask
    else:
        return graph, data_mask


def show_graph_nodes(graph, mesh, data, clim=(0, 1), transl=None):

    graph_no_dummy, data_mask = remove_dummy_nodes(graph)
    # manage nodes
    s_coords = graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', mesh)
    print("s_coords",s_coords.shape)

    transl_bary = np.mean(s_coords)
    s_coords = 1.01*(s_coords-transl_bary)+transl_bary

    if transl is not None:
        s_coords += transl
    print(data_mask.shape)
    print(data.shape)
    #print(data[data_mask].shape)
    data_mask = data_mask[:88]
    print("data_mask:",data[data_mask].shape)

    s_obj = SourceObj('nodes', s_coords, color='red',#data=data[data_mask],
                        edge_color='black', symbol='disc', edge_width=2.,
                        radius_min=30., radius_max=30., alpha=.9)
    """Color the sources according to data
    """
    s_obj.color_sources(data=data[data_mask], cmap='jet', clim=clim)
    # Get the colorbar of the source object
    CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), txtcolor='k')
    cb_obj = ColorbarObj(s_obj, cblabel='node consistency', border=False,
                  **CBAR_STATE)

    return s_obj, cb_obj


def visbrain_plot(mesh, tex=None, caption=None, cblabel=None, visb_sc=None,
                  cmap='jet'):
    """
    Visualize a trimesh object using visbrain core plotting tool
    :param mesh: trimesh object
    :param tex: numpy array of a texture to be visualized on the mesh
    :return:
    """
    from visbrain.objects import BrainObj, ColorbarObj, SceneObj
    b_obj = BrainObj('gui', vertices=np.array(mesh.vertices),
                     faces=np.array(mesh.faces),
                     translucent=False,
                     hemisphere="both")

    #b_obj.rotate(fixed="bottom", scale_factor=0.02)
    if visb_sc is None:
        visb_sc = SceneObj(bgcolor='white', size=(1400, 1000))
        visb_sc.add_to_subplot(b_obj, title=caption)
        visb_sc_shape = (1, 1)
    else:
        visb_sc_shape = get_visb_sc_shape(visb_sc)
        visb_sc.add_to_subplot(b_obj, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1], title=caption)

    if tex is not None:
        b_obj.add_activation(data=tex, cmap=cmap,
                             clim=(np.min(tex), np.max(tex)))
        CBAR_STATE = dict(cbtxtsz=20, txtsz=20., width=.1, cbtxtsh=3.,
                          rect=(-.3, -2., 1., 4.), cblabel=cblabel)
        cbar = ColorbarObj(b_obj, **CBAR_STATE)
        visb_sc.add_to_subplot(cbar, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1] + 1, width_max=200)
    return visb_sc

def load_graphs_in_list(path_to_graphs_folder):
    """
    Return a list of graph loaded from the path
    """
    list_graphs = []
    for file in os.listdir(path_to_graphs_folder):
        if 'graph' in file and 'gpickle' in file:
            graph = nx.read_gpickle(os.path.join(path_to_graphs_folder,file))
            list_graphs.append(graph)

    return list_graphs

def reg_mesh(mesh):
    # flip
    transfo_full = np.array([[-1, 0, 0, 0],[0, -1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
    # Rz(90)
    transfo_full = np.array([[0, -1, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
    # Rx(-90)
    transfo_full = np.array([[1, 0, 0, 0],[0, 0, 1, 0],[0, -1, 0, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
    return mesh


if __name__ == "__main__":
    template_mesh = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs'
    path_to_node_cons = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch'

    list_graphs = load_graphs_in_list(path_to_graphs)
    for g in list_graphs:
        print(len(g))
    pickle_in = open(os.path.join(path_to_node_cons,"nodeCstPerGraph_mALS.pck"),"rb")
    nodeCstPerGraph_mALS = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_node_cons,"nodeCstPerGraph_mSync.pck"),"rb")
    nodeCstPerGraph_mSync = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_node_cons,"nodeCstPerGraph_CAO.pck"),"rb")
    nodeCstPerGraph_CAO = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_node_cons,"nodeCstPerGraph_KerGM.pck"),"rb")
    nodeCstPerGraph_KerGM = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_node_cons,"nodeCstPerGraph_Hippi.pck"),"rb")
    nodeCstPerGraph_Hippi = pickle.load(pickle_in)
    pickle_in.close()


    print("Node consistency mALS:",np.mean(nodeCstPerGraph_mALS))
    print("Node consistency mSync:",np.mean(nodeCstPerGraph_mSync))
    print("Node consistency KerGM:",np.mean(nodeCstPerGraph_KerGM))
    print("Node consistency CAO:",np.mean(nodeCstPerGraph_CAO))
    print("Node consistency Hippi:",np.mean(nodeCstPerGraph_Hippi))

    # Get the mesh
    mesh = reg_mesh(sio.load_mesh(template_mesh))

    # vb_sc = None
    # clim=(0.8, 1)
    # for ind_g, g in enumerate(list_graphs[:4]):
    #     s_obj, cb_obj = show_graph_nodes(g, mesh, data=nodeCstPerGraph_mALS[:,ind_g], clim=clim)
    #     vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    #     visb_sc_shape = get_visb_sc_shape(vb_sc)
    #     vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    # vb_sc = None
    # clim=(0, 1)
    # ind_g=0
    # g=list_graphs[ind_g]
    # s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_KerGM,1), clim=clim)
    # vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    # visb_sc_shape = get_visb_sc_shape(vb_sc)
    # vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    # vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
    #                            col=visb_sc_shape[1] + 1, width_max=60)
    # vb_sc.preview()

    vb_sc = None
    clim=(0.8, 0.95)
    s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_Hippi,1), clim=clim, transl=[0,0,2])
    vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)
    visb_sc_shape = get_visb_sc_shape(vb_sc)
    vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)

    # Ry(180)
    transfo_full = np.array([[-1, 0, 0, 0],[0, 1, 0, 0],[0, 0, -1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
    s_obj, cb_obj = show_graph_nodes(g, mesh, data=np.mean(nodeCstPerGraph_Hippi,1), clim=clim, transl=[0,0,2])
    vb_sc = visbrain_plot(mesh, visb_sc=vb_sc)  
    visb_sc_shape = get_visb_sc_shape(vb_sc)
    vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
    vb_sc.add_to_subplot(cb_obj, row=visb_sc_shape[0] - 1,
                               col=visb_sc_shape[1] + 1, width_max=200)
    vb_sc.preview()

    print(np.min(np.mean(nodeCstPerGraph_Hippi, 1)))
    print(np.max(np.mean(nodeCstPerGraph_Hippi, 1)))

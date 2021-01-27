import networkx as nx
import numpy as np
from visbrain.objects import SourceObj, ConnectObj, ColorbarObj
import tools.graph_processing as gp


def get_visb_sc_shape(visb_sc):
    """
    get the subplot shape in a visbrain scene
    :param visb_sc:
    :return: tuple (number of rows, number of coloumns)
    """
    k = list(visb_sc._grid_desc.keys())
    return k[-1]


def graph_edges_to_connect(graph, edge_attribute=None, nodes_mask=None):

    if edge_attribute is None:
         attr_mat= nx.adjacency_matrix(graph)
         conn_mat = attr_mat.todense()
    else:
        attr_mat = nx.attr_matrix(graph, edge_attr=edge_attribute)
        conn_mat = attr_mat[0]
    if nodes_mask is not None:
        print(nodes_mask.shape)
        conn_mat = np.delete(conn_mat, np.where(nodes_mask==False)[0], 0)
        conn_mat = np.delete(conn_mat, np.where(nodes_mask==False)[0], 1)
        print(conn_mat.shape)
    connect = np.ma.masked_array(np.array(conn_mat), False)
    #connect.mask[np.tril_indices_from(connect.mask)] = True
    return connect


def show_graph(graph, mesh, nodes_color='data', edge_attribute=None, mask_slice_coord=None):

    graph_no_dummy = gp.remove_dummy_nodes(graph)
    # manage nodes
    s_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', mesh)
    if mask_slice_coord is not None:
        nodes_mask = s_coords[:, 2]>mask_slice_coord
        s_coords = s_coords[nodes_mask, :]
    else:
        nodes_mask=None
    connect = graph_edges_to_connect(graph_no_dummy, edge_attribute, nodes_mask)


    if nodes_color=='data':
        data = gp.graph_nodes_attribute(graph_no_dummy, 'depth')
        s_obj = SourceObj('S_left', s_coords, data=data[nodes_mask], color='red',
                            edge_color='black', symbol='disc', edge_width=2.,
                            radius_min=20., radius_max=30., alpha=.4)


        """Color the sources according to data
        """
        s_obj.color_sources(data=data, cmap='plasma')
    else:
        s_obj = SourceObj('S_left', s_coords, color=nodes_color,
                        edge_color='black', symbol='disc', edge_width=2.,
                        radius_min=20., radius_max=30., alpha=.4)

    # manage edges

    #A = nx.adjacency_matrix(graph)
    c_obj = ConnectObj('C_left', s_coords, connect, select=connect>0)
    # c_obj = ConnectObj('C_left', s_coords, connect, color_by='strength',
    #                      cmap='viridis', vmin=0., vmax=.1,
    #                      under='gray', over='red')

    return s_obj, c_obj


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
    mesh.vertices = mesh.vertices - np.mean(mesh.vertices, 0)
    return mesh

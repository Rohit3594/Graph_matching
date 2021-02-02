import networkx as nx
import numpy as np
from visbrain.objects import SourceObj, ConnectObj, ColorbarObj
import tools.graph_processing as gp

CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.), txtcolor='k', border=False)

def get_visb_sc_shape(visb_sc):
    """
    get the subplot shape in a visbrain scene
    :param visb_sc:
    :return: tuple (number of rows, number of coloumns)
    """
    k = list(visb_sc._grid_desc.keys())
    return k[-1]


def graph_nodes_to_sources(graph_no_dummy, nodes_coords, node_color_attribute=None, nodes_mask=None):
    if nodes_mask is None:
        nodes_mask = np.ones((nodes_coords.shape[0],), dtype=np.bool)
    s_obj = SourceObj('nodes', nodes_coords[nodes_mask], color='red',
                        edge_color='black', symbol='disc', edge_width=2.,
                        radius_min=30., radius_max=30., alpha=.9)

    """Color the sources according to data
    """
    data = gp.graph_nodes_attribute(graph_no_dummy, node_color_attribute)
    if len(data) > 0:
        s_obj.color_sources(data=data[nodes_mask], cmap='jet')
        # Get the colorbar of the source object
        cb_obj = ColorbarObj(s_obj, cblabel='node attribute : '+node_color_attribute, **CBAR_STATE)
    else:
        s_obj = SourceObj('nodes', nodes_coords[nodes_mask], color='red',
                        edge_color='black', symbol='disc', edge_width=2.,
                        radius_min=20., radius_max=30., alpha=.4)
        cb_obj = None
    return s_obj, cb_obj


def graph_edges_to_connect(graph, nodes_coords, edge_attribute=None, nodes_mask=None):

    if edge_attribute is None:
         attr_mat= nx.adjacency_matrix(graph)
         conn_mat = attr_mat.todense()
    else:
        attr_mat = nx.attr_matrix(graph, edge_attr=edge_attribute)
        conn_mat = attr_mat[0]
    if nodes_mask is not None:
        conn_mat = np.delete(conn_mat, np.where(nodes_mask==False)[0], 0)
        conn_mat = np.delete(conn_mat, np.where(nodes_mask==False)[0], 1)
    connect = np.ma.masked_array(np.array(conn_mat), False)
    if nodes_mask is not None:
        c_obj = ConnectObj('edges', nodes_coords[nodes_mask], connect, select=connect>0, cmap='viridis')
    else:
        c_obj = ConnectObj('edges', nodes_coords, connect, select=connect>0, cmap='viridis')

    # c_obj = ConnectObj('edges', s_coords, connect, color_by='strength',
    #                      cmap='viridis', vmin=0., vmax=.1,
    #                      under='gray', over='red')

    return c_obj


def show_graph(graph_no_dummy, nodes_coords, node_color_attribute=None, edge_color_attribute=None, nodes_mask=None):

    # manage nodes
    s_obj, nodes_cb_obj = graph_nodes_to_sources(graph_no_dummy, nodes_coords, node_color_attribute, nodes_mask)

    # manage edges
    c_obj = graph_edges_to_connect(graph_no_dummy, nodes_coords, edge_color_attribute, nodes_mask)

    return s_obj, c_obj, nodes_cb_obj


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
        # cbar = ColorbarObj(b_obj, cblabel=cblabel, **CBAR_STATE)
        # visb_sc.add_to_subplot(cbar, row=visb_sc_shape[0] - 1,
        #                        col=visb_sc_shape[1] + 1, width_max=200)
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

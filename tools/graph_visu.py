import networkx as nx
import numpy as np
from visbrain.objects import SourceObj, ConnectObj, ColorbarObj
import tools.graph_processing as gp
import slam.differential_geometry as sdg
import slam.plot as splt

CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.), txtcolor='k', border=False)


def graph_nodes_coords_to_sources(graph_no_dummy):
    nodes_coords = gp.graph_nodes_attribute(graph_no_dummy, 'sphere_3dcoords')
    s_obj = SourceObj('nodes', nodes_coords, color='black', symbol='o',
                        radius_min=15., radius_max=15., alpha=.7)
    return s_obj


def nodes_density_map(list_graphs, mesh, nb_iter=10, dt=0.5):
    """
    Return the smoothed texture of all non labeled points
    """

    mesh_size = mesh.vertices.shape[0]

    # initialise texture
    non_smoothed_texture = np.zeros(mesh_size)

    for graph in list_graphs:
        for node in graph.nodes:
            non_smoothed_texture[graph.nodes[node]["ico100_7_vertex_index"]] += 1

    # normalization with respect to the number of graphs
    non_smoothed_texture = non_smoothed_texture/len(list_graphs)

    # smooth the texture
    # smoothed_texture = sdg.laplacian_texture_smoothing(mesh,
    #                                                    non_smoothed_texture,
    #                                                    nb_iter,
    #                                                    dt)

    return non_smoothed_texture


def get_visb_sc_shape(visb_sc):
    """
    get the subplot shape in a visbrain scene
    :param visb_sc:
    :return: tuple (number of rows, number of coloumns)
    """
    k = list(visb_sc._grid_desc.keys())
    return k[-1]


def graph_nodes_to_sources(nodes_coords, node_data=None, nodes_size=None, nodes_mask=None, c_map=None):
    if nodes_size is None:
        nodes_size = 15.

    # dilate a bit the coords to make the circles correspoding to sources more visible
    transl_bary = np.mean(nodes_coords)
    nodes_coords = 1.01*(nodes_coords-transl_bary)+transl_bary



    # apply the mask if provided
    if nodes_mask is None:
        nodes_mask = np.ones((nodes_coords.shape[0],), dtype=np.bool)
    s_obj = SourceObj('nodes', nodes_coords[nodes_mask], color='black',
                        edge_color='black', symbol='o', edge_width=2.,
                        radius_min=nodes_size, radius_max=nodes_size, alpha=.7)

    """Color the sources according to data
    """    
    if node_data is not None:
        if c_map is None:
            c_map = 'jet'
        s_obj.color_sources(data=node_data[nodes_mask], cmap=c_map)
        # Get the colorbar of the source object
        cb_obj = ColorbarObj(s_obj, **CBAR_STATE)

    else:
        s_obj = SourceObj('nodes', nodes_coords[nodes_mask], color='purple',
                        edge_color='black', symbol='o', edge_width=2.,
                        radius_min=nodes_size, radius_max=nodes_size, alpha=.4)
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
        c_obj = ConnectObj('edges', nodes_coords[nodes_mask], connect, select=connect>0, cmap='inferno')
    else:
        c_obj = ConnectObj('edges', nodes_coords, connect, select=connect>0, cmap='inferno')

    # c_obj = ConnectObj('edges', nodes_coords, connect, color_by='strength',
    #                      cmap='viridis', vmin=0., vmax=.1,
    #                      under='gray', over='red')

    return c_obj


def graph_edges_select(graph, nodes_coords, edge_attribute, attribute_threshold):

    attr_mat = nx.attr_matrix(graph, edge_attr=edge_attribute)
    conn_mat = attr_mat[0]

    connect = np.ma.masked_array(np.array(conn_mat), False)
    c_obj = ConnectObj('edges', nodes_coords, connect, select=connect>attribute_threshold, cmap='viridis')

    return c_obj


def show_graph(graph_no_dummy, nodes_coords, node_color_attribute=None, edge_color_attribute=None, nodes_size=None, nodes_mask=None, c_map=None):

    # manage nodes
    if node_color_attribute is not None:
        node_data = gp.graph_nodes_attribute(graph_no_dummy, node_color_attribute)
    else:
        node_data = None
    s_obj, nodes_cb_obj = graph_nodes_to_sources(nodes_coords, node_data=node_data, nodes_size=nodes_size, nodes_mask=nodes_mask, c_map=c_map)

    # manage edges
    c_obj = graph_edges_to_connect(graph_no_dummy, nodes_coords, edge_color_attribute, nodes_mask)

    return s_obj, c_obj, nodes_cb_obj


def visbrain_plot(mesh, tex=None, caption=None, cblabel=None, visb_sc=None,
                  cmap='jet',clim=None):
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

        if clim is not None:
            b_obj.add_activation(data=tex, cmap=cmap,clim=clim)

        else:
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

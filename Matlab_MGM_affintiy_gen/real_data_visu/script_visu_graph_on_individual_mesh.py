import os
import slam.io as sio
import slam.topology as stop
import slam.plot as splt
import slam.mapping as smap
import slam.differential_geometry as sdg
import numpy as np
import networkx as nx
import argparse
import pickle

from visbrain.gui import Brain
from visbrain.objects import SourceObj, ConnectObj
from visbrain.io import download_file


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


def graph_nodes_to_coords(graph, index_attribute, mesh):
    vert_indices = list(nx.get_node_attributes(graph, index_attribute).values())
    coords = np.array(mesh.vertices[vert_indices, :])
    return coords


def graph_nodes_attribute(graph, attribute):
    att = list(nx.get_node_attributes(graph, attribute).values())
    return np.array(att)


def remove_dummy_nodes(graph):
    is_dummy = graph_nodes_attribute(graph, 'is_dummy')
    if True in is_dummy:
        graph_copy = graph.copy()
        graph_copy.remove_nodes_from(np.where(np.array(is_dummy)==True)[0])
        return graph_copy
    else:
        return graph


def show_graph(graph, mesh, nodes_color='data', edge_attribute=None, mask_slice_coord=None, transl=None):

    graph_no_dummy = remove_dummy_nodes(graph)
    # manage nodes
    s_coords = graph_nodes_to_coords(graph_no_dummy, 'vertex_index', mesh)
    if mask_slice_coord is not None:
        nodes_mask = s_coords[:, 2]>mask_slice_coord
        s_coords = s_coords[nodes_mask, :]
    else:
        nodes_mask=None
    if transl is not None:
        s_coords += transl
    connect = graph_edges_to_connect(graph_no_dummy, edge_attribute, nodes_mask)


    if nodes_color=='data':
        data =graph_nodes_attribute(graph_no_dummy, 'depth')
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
    c_obj = ConnectObj('C_left', s_coords, connect, select=connect>0,line_width=4,
                       antialias=True, color_by='strength', vmin=0., vmax=.5,over='black')
    # c_obj = ConnectObj('C_left', s_coords, connect, color_by='strength',
    #                      cmap='viridis', vmin=0., vmax=.1,
    #                      under='gray', over='red')

    return s_obj, c_obj


def visbrain_plot(mesh, tex=None, caption=None, cblabel=None, visb_sc=None,
                  cmap='jet', colbar=False):
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
        if colbar:
            CBAR_STATE = dict(cbtxtsz=20, txtsz=20., width=.1, cbtxtsh=3.,
                              rect=(-.3, -2., 1., 4.), cblabel=cblabel)
            cbar = ColorbarObj(b_obj, **CBAR_STATE)
            visb_sc.add_to_subplot(cbar, row=visb_sc_shape[0] - 1,
                                   col=visb_sc_shape[1] + 1, width_max=200)
    return visb_sc


def load_graphs_in_list(path):
    """
    Return a list of graph loaded from the path
    """
    path_to_graphs_folder = os.path.join(path, "modified_graphs")
    list_graphs = []
    for i_graph in range(0,len(os.listdir(path_to_graphs_folder))):
        path_graph = os.path.join(path_to_graphs_folder, "graph_"+str(i_graph)+".gpickle")
        graph = nx.read_gpickle(path_graph)
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
        # Rz(90)
    transfo_full = np.array([[0, -1, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
        # Rz(90)
    transfo_full = np.array([[0, -1, 0, 0],[1, 0, 0, 0],[0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(transfo_full)
    mesh.vertices = mesh.vertices - np.mean(mesh.vertices, 0)
    return mesh


if __name__ == "__main__":
    file_mesh = '/mnt/data/work/python_sandBox/stage_nathan/data/OASIS_0061_for_visu/rh.white.gii'
    file_basins = '/mnt/data/work/python_sandBox/stage_nathan/data/OASIS_0061_for_visu/alpha0.03_an0_dn20_r1.5_R_area50FilteredTexture.gii'
    file_graph = '/mnt/data/work/python_sandBox/stage_nathan/data/OASIS_0061_for_visu/OAS1_0061_rh_pitgraph.gpickle'

    graph = nx.read_gpickle(file_graph)
    # Get the mesh
    mesh = reg_mesh(sio.load_mesh(file_mesh))
    import trimesh.smoothing as tms
    mesh = tms.filter_laplacian(mesh,iterations=80)
    #mesh.show()
    tex_basins = sio.load_texture(file_basins)
    vb_sc = visbrain_plot(mesh,tex=tex_basins.darray[2], cmap='tab20c' )
    s_obj, c_obj = show_graph(graph, mesh, 'red', edge_attribute=None, mask_slice_coord=-15, transl=[0,0,2])
    vb_sc.add_to_subplot(c_obj)
    vb_sc.add_to_subplot(s_obj)


    vb_sc.preview()

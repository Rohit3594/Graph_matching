import slam.io as sio
import networkx as nx
import numpy as np
import tools.graph_visu as gv
import tools.graph_processing as gp
from visbrain.objects import SourceObj, ConnectObj, ColorbarObj


def graph_edges_select(graph, nodes_coords, edge_attribute, attribute_threshold):

    attr_mat = nx.attr_matrix(graph, edge_attr=edge_attribute)
    conn_mat = attr_mat[0]

    connect = np.ma.masked_array(np.array(conn_mat), False)
    c_obj = ConnectObj('edges', nodes_coords, connect, select=connect>attribute_threshold, cmap='viridis')

    return c_obj


if __name__ == "__main__":
    file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    file_graph = '../data/example_individual_OASIS_0061/OAS1_0061_rh_pitgraph.gpickle'
    file_mesh = '../data/example_individual_OASIS_0061/rh.white.gii'
    file_basins = '../data/example_individual_OASIS_0061/alpha0.03_an0_dn20_r1.5_R_area50FilteredTexture.gii'

    graph = nx.read_gpickle(file_graph)
    gp.preprocess_graph(graph)

    edge_geo_dist = gp.graph_edges_attribute(graph, 'geodesic_distance')
    print(np.min(edge_geo_dist))
    print(np.max(edge_geo_dist))
    print(np.mean(edge_geo_dist))

    mesh = sio.load_mesh(file_mesh)
    # eventually smooth it a bit
    import trimesh.smoothing as tms
    mesh = tms.filter_laplacian(mesh, iterations=300)
    # load the basins texture
    tex_basins = sio.load_texture(file_basins)
    # plot the mesh with basin texture
    vb_sc = gv.visbrain_plot(mesh, tex=tex_basins.darray[2], cmap='tab20c',
                             caption='Visu on individual mesh with basins',)
    gp.remove_dummy_nodes(graph)
    nodes_coords = gp.graph_nodes_to_coords(graph, 'vertex_index', mesh)
    # manage nodes
    s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(graph, nodes_coords)
    # manage edges
    c_obj = graph_edges_select(graph, nodes_coords, edge_attribute='geodesic_distance', attribute_threshold=80)

    vb_sc.add_to_subplot(c_obj)
    vb_sc.add_to_subplot(s_obj)

    # show the plot on the screen
    vb_sc.preview()




    # # load and reorient the template mesh
    # template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))
    #
    # # plot the mesh with basin texture
    # vb_sc = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
    # # process the graph
    # # 1 remove potential dummy nodes (which are not connected by any edge and have no coordinate)
    # gp.remove_dummy_nodes(graph)
    # # 2 compute nodes coordinates in 3D by retrieving the mesh vertex corresponding to each graph node, based on the
    # # corresponding node attribute
    # nodes_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', template_mesh)
    #
    # # manage nodes
    # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(graph, nodes_coords)
    #
    # # manage edges
    # c_obj = graph_edges_select(graph, nodes_coords, edge_attribute='geodesic_distance', attribute_threshold=80)
    #
    #
    # vb_sc.add_to_subplot(c_obj)
    # vb_sc.add_to_subplot(s_obj)
    #
    # # show the plot on the screen
    # vb_sc.preview()
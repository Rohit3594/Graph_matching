import sys
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p

def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
    nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)
    one_nodes_coords = nodes_coords[:, coord_dim]
    one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords))/(np.max(one_nodes_coords)-np.min(one_nodes_coords))
    # initialise the dict for atttributes
    nodes_attributes = {}
    # Fill the dictionnary with the nd_array attribute
    for ind, node in enumerate(graph_no_dummy.nodes):
        nodes_attributes[node] = {"label_color": one_nodes_coords_scaled[ind]}

    nx.set_node_attributes(graph_no_dummy, nodes_attributes)

if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/OASIS_full_batch/modified_graphs'

    path_to_match_mat = "../data/OASIS_full_batch/X_mALS.mat"

    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    x_mSync = sco.loadmat(path_to_match_mat)['X']

    matching_matrix = x_mSync
    nb_graphs = 134

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


    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for g in list_graphs:
        gp.remove_dummy_nodes(g)
        print(len(g))

    # Get the mesh
    mesh = sio.load_mesh(template_mesh)
    vb_sc = gv.visbrain_plot(mesh)
    # gp.remove_dummy_nodes(g)
    # label_nodes_according_to_coord(g, mesh, coord_dim=1)
    # nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
    # s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_color_attribute="label_color", nodes_mask=None, c_map='nipy_spectral')#'rainbow')
    # vb_sc.add_to_subplot(s_obj)
    # vb_sc.preview()

    for ind_g, g in enumerate(list_graphs):
        gp.remove_dummy_nodes(g)
        label_nodes_according_to_coord(g, mesh, coord_dim=1)
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', mesh)
        node_data = gp.graph_nodes_attribute(g, "label_color")
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(g, nodes_coords, node_data=node_data, nodes_mask=None, c_map='nipy_spectral')#'rainbow')
        vb_sc.add_to_subplot(s_obj)

    vb_sc.preview()
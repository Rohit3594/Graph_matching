import sys
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import os
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p
import copy


if __name__ == "__main__":
    #template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/mnt/data/work/python_sandBox/Graph_matching/data/OASIS_labelled_pits_graphs'
    #path_to_match_mat = "/home/rohit/PhD_Work/GM_my_version/RESULT_FRIOUL_HIPPI/Hippi_res_real_mat.npy"

    list_graphs = gp.load_labelled_graphs_in_list(path_to_graphs, hemi='lh')
    mesh = sio.load_mesh(template_mesh)
    reg_mesh = gv.reg_mesh(mesh)
    vb_sc = gv.visbrain_plot(reg_mesh)

    for g in list_graphs:
        nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', reg_mesh)
        #labels = nx.get_node_attributes(g, 'label_media').values()
        labels = nx.get_node_attributes(g, 'label_neuroimage').values()
        color_label = np.array([l/100 for l in labels])
        #print(color_label)
        s_obj, nodes_cb_obj = gv.graph_nodes_to_sources(nodes_coords, node_data=color_label, nodes_mask=None, c_map='nipy_spectral')
        vb_sc.add_to_subplot(s_obj)

    vb_sc.preview()

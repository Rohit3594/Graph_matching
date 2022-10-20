import sys
import os
#sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp
import tools.clusters_analysis as gca
import numpy as np
import networkx as nx
import scipy.io as sco
import pickle as p


if __name__ == "__main__":
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_match_mat = '../data/Oasis_original_new_with_dummy/'
    method = 'neuroimage'#'media'#'CAO'#'kerGM'#'mSync'#'mALS'#
    path_to_X = "../data/Oasis_original_new_with_dummy/X_"+method+".mat"
    X = sco.loadmat(path_to_X)['full_assignment_mat']
    print(X.shape)
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    X_med = gca.get_assignment_from_labelling(list_graphs, labelling_attribute_name='label_'+method)
    # for i, g in enumerate(list_graphs):
    #     gp.remove_dummy_nodes(g)

    print(np.max(X-X_med))
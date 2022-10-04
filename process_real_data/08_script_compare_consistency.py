import os
import slam.io as sio
import numpy as np
import networkx as nx
import pickle
from visbrain.objects import SourceObj, ColorbarObj
import tools.graph_visu as gv
import tools.graph_processing as gp

if __name__ == "__main__":
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs'
    path_to_consistency = '../data/Oasis_original_new_with_dummy/consistency'
    path_to_figs = '../data/Oasis_original_new_with_dummy/figures'
    reg_or_unreg = ''#'_unreg'#''


    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_mALS"+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph_mALS = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_mSync"+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph_mSync = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_CAO.pck"),"rb")
    nodeCstPerGraph_CAO = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_KerGM"+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph_KerGM = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_kmeans_110_real_data_dummy"+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph_kmeans_110 = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_kmeans_90_real_data_dummy"+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph_kmeans_90 = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_kmeans_70_real_data_dummy"+reg_or_unreg+".pck"),"rb")
    nodeCstPerGraph_kmeans_70 = pickle.load(pickle_in)
    pickle_in.close()
    # pickle_in = open(os.path.join(path_to_consistency,"nodeCstPerGraph_Hippi.pck"),"rb")
    # nodeCstPerGraph_Hippi = pickle.load(pickle_in)
    # pickle_in.close()

    print("Node consistency mALS:", np.mean(nodeCstPerGraph_mALS), np.std(nodeCstPerGraph_mALS))
    print("Node consistency mSync:", np.mean(nodeCstPerGraph_mSync), np.std(nodeCstPerGraph_mSync))
    print("Node consistency KerGM:", np.mean(nodeCstPerGraph_KerGM), np.std(nodeCstPerGraph_KerGM))
    print("Node consistency CAO:", np.mean(nodeCstPerGraph_CAO), np.std(nodeCstPerGraph_CAO))
    print("Node consistency kmeans110:", np.mean(nodeCstPerGraph_kmeans_110), np.std(nodeCstPerGraph_kmeans_110))
    print("Node consistency kmeans90:", np.mean(nodeCstPerGraph_kmeans_90), np.std(nodeCstPerGraph_kmeans_90))
    print("Node consistency kmeans70:", np.mean(nodeCstPerGraph_kmeans_70), np.std(nodeCstPerGraph_kmeans_70))

    # print("Node consistency Hippi:",np.mean(nodeCstPerGraph_Hippi), np.std(nodeCstPerGraph_Hippi))

    print(np.mean(nodeCstPerGraph_mALS,1))
    print(np.std(nodeCstPerGraph_mALS,1))
    #print(np.mean(nodeCstPerGraph_mSync,1))
    #print(np.mean(nodeCstPerGraph_KerGM,1))
    #print(np.mean(nodeCstPerGraph_CAO,1))
    #rank_mSync = np.linalg.matrix_rank(matching_mSync)
    #print(rank_mSync)

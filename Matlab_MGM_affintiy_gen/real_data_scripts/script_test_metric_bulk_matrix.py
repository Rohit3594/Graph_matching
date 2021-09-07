import scipy.io as sio
import os
import numpy as np
import pickle

def compute_node_consistency(matching_matrix, nb_graphs, nb_nodes):
    nodeCstPerGraph = np.zeros((nb_nodes, nb_graphs))
    for graph_ref_num in range(nb_graphs):
        print('graph_ref_num=', graph_ref_num)
        #rscope = (graph_ref_num - 1) * nb_nodes + 1:graph_ref_num * nb_nodes
        rscope = range(graph_ref_num * nb_nodes, (graph_ref_num + 1) * nb_nodes)
        for i in range(nb_graphs-1):
            #x_k_i = matching_matrix[, ]
            iscope = range(i * nb_nodes, (i+1)*nb_nodes)
            Xri = np.array(matching_matrix[np.ix_(rscope, iscope)], dtype=int)
            for j in range(i+1,nb_graphs):
                jscope = range(j * nb_nodes, (j + 1) * nb_nodes)
                Xij = np.array(matching_matrix[np.ix_(iscope, jscope)], dtype=int)
                Xrj = np.array(matching_matrix[np.ix_(rscope, jscope)], dtype=int)
                Xrij = np.matmul(Xri, Xij)
                nodeCstPerGraph[:, graph_ref_num] += (1-np.sum(np.abs(Xrij-Xrj),1)/2)

    # normalize the summation value
    nodeCstPerGraph = nodeCstPerGraph/(nb_graphs*(nb_graphs-1)/2)
    # sort
    # [~,IX] = np.sort(nodeCstPerGraph,1,'descend')
    # nodeCstPerGraph2 = np.zeros(nb_nodes,nb_graphs)
    # for ref in range(nb_graphs):
    #     nodeCstPerGraph2(IX(1:inCnt,ref),ref) = 1
    # nodeCstPerGraph = nodeCstPerGraph2
    return nodeCstPerGraph


if __name__ == "__main__":


    #path_to_read = '/hpc/meca/users/buskulic.n/stage_nathan/data_pits_graph/full_batch'
    # Get the number of graphs by looking into the modified_graph folder
    #nb_graphs = len(os.listdir(os.path.join(path_to_read, "modified_graphs")))
    path_to_read = '/mnt/data/work/python_sandBox/stage_nathan/data/OASIS_full_batch'
    nb_graphs = 134
    # load the mALS results
    matching_mALS = sio.loadmat(os.path.join(path_to_read,"X_mALS.mat"))["X"]
    matching_mSync = sio.loadmat(os.path.join(path_to_read,"X_mSync.mat"))["X"]
    matching_pairwise = sio.loadmat(os.path.join(path_to_read,"X_pairwise_kergm.mat"))["full_assignment_mat"]

    # get the associated number of nodes
    nb_nodes = int(matching_mALS.shape[0]/nb_graphs)
    nodeCstPerGraph_mALS = compute_node_consistency(matching_mALS, nb_graphs, nb_nodes)
    nodeCstPerGraph_mSync = compute_node_consistency(matching_mSync, nb_graphs, nb_nodes)
    nodeCstPerGraph_KerGM = compute_node_consistency(matching_pairwise, nb_graphs, nb_nodes)

    pickle_out = open(os.path.join(path_to_read,"nodeCstPerGraph_mALS.pck"),"wb")
    pickle.dump(nodeCstPerGraph_mALS, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_read,"nodeCstPerGraph_mSync.pck"),"wb")
    pickle.dump(nodeCstPerGraph_mSync, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join(path_to_read,"nodeCstPerGraph_KerGM.pck"),"wb")
    pickle.dump(nodeCstPerGraph_KerGM, pickle_out)
    pickle_out.close()
    print(np.mean(nodeCstPerGraph_mALS))
    print(np.mean(nodeCstPerGraph_mSync))
    print(np.mean(nodeCstPerGraph_KerGM))


    print(np.mean(nodeCstPerGraph_mALS,1))
    print(np.std(nodeCstPerGraph_mALS,1))
    print(np.mean(nodeCstPerGraph_mSync,1))
    print(np.mean(nodeCstPerGraph_KerGM,1))
    #rank_mSync = np.linalg.matrix_rank(matching_mSync)
    #print(rank_mSync)
import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_generation.load_graphs_and_create_metadata import dataset_metadata
from graph_matching_tools.metrics import matching
import matplotlib.pyplot as plt
import scipy.io as sio
import tools.graph_processing as gp
import scipy.stats as stats

path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/labelled_graphs'
path_ro_correspondence = '../Matlab_MGM_affintiy_gen/gender_correspondence.pickle'

labeled_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)
gender_corresp = np.array(nx.read_gpickle(path_ro_correspondence))[:,2] # gender correp list

def create_clusters_lists_with_label_gender(list_graphs,gender_corresp,label_attribute="label_dbscan"):

    result_dict = {}
    label_depths = {}
    label_gender = {}

    for i_graph, graph in enumerate(list_graphs):
        for node in graph.nodes:
            if not graph.nodes[node]["is_dummy"]:
                label_cluster = graph.nodes[node][label_attribute]
                
                if label_cluster in result_dict:
                    
                    #retrieve depth of the corresponding label in that graph
                    depth_value = graph.nodes[node]['depth']
                    
                    result_dict[label_cluster].append((i_graph, node))
                    label_depths[label_cluster].append(depth_value)
                    label_gender[label_cluster].append(gender_corresp[i_graph])
                    
                else:
                    #retrieve depth of the corresponding label in that graph
                    depth_value = graph.nodes[node]['depth']
                    
                    result_dict[label_cluster] = [(i_graph, node)]
                    label_depths[label_cluster] = [depth_value]
                    label_gender[label_cluster] = [gender_corresp[i_graph]]


    return result_dict,label_depths,label_gender


def seperate_groups_by_label(label_gender,label_depths):
    
    # Separate groups by label
    
    label_gen_sep = []
    for key in label_gender.keys():
        M = []
        F = []
        for i in range(len(label_gender[key])):

            if label_gender[key][i] == 'F':

                F.append(label_depths[key][i])
            else:
                M.append(label_depths[key][i])
                
         # 1st list M, 2nd F
        
        label_gen_sep.append([M,F])
        
    return label_gen_sep


def calculate_tstats_and_pvalues(corresp ,method = 'labelling_mALS'):
    
    # get labeled groups and depths    
    result_dict,label_depths,label_gender = create_clusters_lists_with_label_gender(labeled_graphs,corresp,method)
    
    # depth seperated by groups
    label_gen_sep = seperate_groups_by_label(label_gender,label_depths)
    
    t_stats = {}

    for key,lst in zip(label_gender.keys(),label_gen_sep):

        res = stats.ttest_ind(a=lst[0], b=lst[1], equal_var=True)

        t_stats[key] = [res[0],res[1]]
        
    return t_stats


if __name__ == '__main__':

	tstats_mALS = calculate_tstats_and_pvalues(gender_corresp, 'labelling_mALS')

	tstats_mSync = calculate_tstats_and_pvalues(gender_corresp, 'labelling_mSync')

	tstats_matcheig = calculate_tstats_and_pvalues(gender_corresp, 'labelling_MatchEig')

	tstats_CAO = calculate_tstats_and_pvalues(gender_corresp, 'labelling_CAO')

	tstats_kerGM = calculate_tstats_and_pvalues(gender_corresp, 'labelling_kerGM')


	print('tstats_mALS : \n', tstats_mALS)



import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_generation.load_graphs_and_create_metadata import dataset_metadata
from graph_matching_tools.metrics import matching
import matplotlib.pyplot as plt
import scipy.io as sco
import slam.io as sio
from scipy.special import softmax
import pickle
from scipy.stats import betabinom
import seaborn as sns
import tools.graph_processing as gp
import tools.graph_visu as gv
from matplotlib.pyplot import figure
import pickle
import pandas as pd
from torch_geometric.utils.convert import from_networkx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.nn.functional import one_hot
from sklearn.preprocessing import OneHotEncoder
from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import TopKPooling
from torch_geometric.data import Data



# file_template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii' #updated new avg template
file_template_mesh = '../data/HCP/Q1-Q6_RelatedValidation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii' 


# file_mesh = '../data/example_individual_OASIS_0061/lh.white.gii'

# path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/modified_graphs/'

path_to_labelled_graphs =  '../data/HCP/modified_graphs_left/'

template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)

#topk_scores = pickle.load(open( "topk_scores_all_7.pickle", "rb" ))
topk_scores = pickle.load(open( "HCP_topk_scores_pool1_2.pickle", "rb" ))

mask_list = pickle.load(open( "HCP_mask_fold_2.pickle", "rb" ))

template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

graph_num = 0

vb_sc1 = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')

# assign topk score as attributes
for g,topk_g in zip(list_graphs,topk_scores):
    
    gp.remove_dummy_nodes(g)
    
    topk_dict = {}
    
    for n,s in zip(g,topk_g):
        topk_dict[n] = {'topk_score':s}
        
    nx.set_node_attributes(g, topk_dict)


graph = list_graphs[graph_num]

print(list_graphs[0].nodes.data()[0])

nodes_coords = gp.graph_nodes_to_coords(graph, 'Glasser2016_vertex_index', template_mesh)

s_obj1, c_obj1, node_cb_obj1 = gv.show_graph(graph, nodes_coords, node_color_attribute='topk_score',
											edge_color_attribute=None,c_map='jet')



vb_sc1.add_to_subplot(s_obj1)
visb_sc_shape1 = gv.get_visb_sc_shape(vb_sc1)
vb_sc1.add_to_subplot(node_cb_obj1, row=visb_sc_shape1[0] - 1, col=visb_sc_shape1[0] + 0, width_max=300)

vb_sc1.preview()



# Plot all the nodes on the template 
vb_sc2 = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
visb_sc_shape2 = gv.get_visb_sc_shape(vb_sc2)


for i,g in enumerate(list_graphs):

	gp.remove_dummy_nodes(g)
	print('graph_num: ',i)

	#gp.sphere_nearest_neighbor_interpolation(g, template_mesh)
	nodes_coords = gp.graph_nodes_to_coords(g, 'Glasser2016_vertex_index', template_mesh)
	s_obj2, c_obj2, node_cb_obj2 = gv.show_graph(g, nodes_coords,node_color_attribute='topk_score',nodes_mask = mask_list[i], nodes_size=8, c_map='jet')
	#vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
	vb_sc2.add_to_subplot(s_obj2)
vb_sc2.add_to_subplot(node_cb_obj2, row=visb_sc_shape2[0] - 1, col=visb_sc_shape2[0] + 0, width_max=300)

vb_sc2.preview()




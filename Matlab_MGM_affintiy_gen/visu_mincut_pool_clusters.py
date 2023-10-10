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



file_template_mesh = '../data/HCP/Q1-Q6_RelatedValidation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii' 

#file_template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii'

path_to_labelled_graphs =  '../data/HCP/modified_graphs_left/'

template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)


# fold = 7
# model.load_state_dict(torch.load('OASIS_MINcut_gender_cross_val_'+str(fold)+'.model'))

clusters_numpy  = pickle.load(open( "./mincut_clusters_with_3Dcoords/HCP/clusters_numpy_fold_HCP7.pickle", "rb" ))



graph_count = 0

for g in clusters_numpy:
    
    dict_lab_mincut = {}
    
    for node,n in enumerate(g):
        dict_lab_mincut[node] = {"mincut_label":np.argmax(n)}
        
    print('Graph count: ',graph_count)

    graph = list_graphs[graph_count]
  
    nx.set_node_attributes(graph,dict_lab_mincut)

    graph_count += 1


vb_sc = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
visb_sc_shape = gv.get_visb_sc_shape(vb_sc)


for i,g in enumerate(list_graphs):
	gp.remove_dummy_nodes(g)
	print('graph_num: ',i)

	#gp.sphere_nearest_neighbor_interpolation(g, template_mesh)
	nodes_coords = gp.graph_nodes_to_coords(g, 'Glasser2016_vertex_index', template_mesh)
	# nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', template_mesh)
	s_obj, c_obj, node_cb_obj = gv.show_graph(g, nodes_coords,node_color_attribute='mincut_label', nodes_size=5, c_map='tab10')
	#vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
	vb_sc.add_to_subplot(s_obj)

vb_sc.add_to_subplot(node_cb_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=300)

vb_sc.preview()




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



file_template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii' #updated new avg template

file_mesh = '../data/example_individual_OASIS_0061/lh.white.gii'

path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/labelled_graphs/'

list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)

kmeans_labels_emb = pickle.load(open( "kmeans_labels_emb.pickle", "rb" ))

template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))


# Extract kmeans_lab on node embeddings for each graph
last_idx = 0
kmeans_lab_all=[]

for g in list_graphs:

	gp.remove_dummy_nodes(g)
	
	size_g = nx.number_of_nodes(g)
	kmean_labels = kmeans_labels_emb[last_idx : last_idx + size_g]  # extract graph wise labels
	
	last_idx += size_g
	
	dict_kmeans_lab = dict(enumerate(kmean_labels))  # Convert to dictionary
	
	nx.set_node_attributes(g, dict_kmeans_lab, 'kmeans_lab_emb')  # Add to graph



# Plot all the nodes on the template 
vb_sc2 = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
visb_sc_shape2 = gv.get_visb_sc_shape(vb_sc2)


for i,g in enumerate(list_graphs):

	gp.remove_dummy_nodes(g)
	print('graph_num: ',i)

	#gp.sphere_nearest_neighbor_interpolation(g, template_mesh)
	nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', template_mesh)
	s_obj2, c_obj2, node_cb_obj2 = gv.show_graph(g, nodes_coords,node_color_attribute='kmeans_lab_emb', nodes_size=10, c_map='inferno')
	#vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
	vb_sc2.add_to_subplot(s_obj2)
	vb_sc2.add_to_subplot(node_cb_obj2, row=visb_sc_shape2[0] - 1, col=visb_sc_shape2[0] + 0, width_max=300)

vb_sc2.preview()




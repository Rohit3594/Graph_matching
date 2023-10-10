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
import slam.plot as splt


#file_template_mesh = '../data/template_mesh/OASIS_avg.lh.white.talairach.reg.ico7.inflated.gii' #updated new avg template
file_template_mesh = '../data/HCP/Q1-Q6_RelatedValidation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii'


# file_mesh = '../data/example_individual_OASIS_0061/lh.white.gii'


#path_to_labelled_graphs = '../data/Oasis_original_new_with_dummy/labelled_graphs/'
path_to_labelled_graphs =  '../data/HCP/modified_graphs_left/'



list_graphs = gp.load_graphs_in_list(path_to_labelled_graphs)
attn_mat_all = pickle.load(open( "HCP_attn_mat_list_layer_3.pickle", "rb" ))



for i,g in enumerate(list_graphs):

	attn_mat = attn_mat_all[i]
	edge_attn_dict = {}
	
	for edge in g.edges:
		
		edge_attn_dict[edge] = {"attn_weight":attn_mat[edge[0],edge[1]]}
	
	nx.set_edge_attributes(g, edge_attn_dict)


graph_num = 0


graph = list_graphs[graph_num]


template_mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

#vb_sc = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')

gp.remove_dummy_nodes(graph)
# 2 compute nodes coordinates in 3D by retrieving the mesh vertex corresponding to each graph node, based on the
# corresponding node attribute
nodes_coords = gp.graph_nodes_to_coords(graph, 'Glasser2016_vertex_index', template_mesh)
# 3 eventually compute a mask for masking some part of the graph
#mask_slice_coord = -15
#nodes_mask = nodes_coords[:, 2] > mask_slice_coord
# 4 create the objects for visualization of the graph and add these to the figure

# s_obj, c_obj, node_cb_obj = gv.show_graph(graph, nodes_coords, node_color_attribute=None,
# 											edge_color_attribute='attn_weight',c_map='jet')



# c_obj, edge_cb_obj = gv.graph_edges_select(graph, nodes_coords, 'attn_weight', 0.5)  # threshold edges with higher attention weights

# vb_sc.add_to_subplot(c_obj)
# vb_sc.add_to_subplot(s_obj)
# visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
# vb_sc.add_to_subplot(edge_cb_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[0] + 0, width_max=300)

# # show the plot on the screen
# vb_sc.preview()



# Create average attention attributes
average_attn = {}

for g in list_graphs:
    
    gp.remove_dummy_nodes(g)
    
    for node in g.nodes:
        
        avg_attn_node = []
        
        for edge in g.edges(node):

        	avg_attn_node.append(g.get_edge_data(edge[0],edge[1])['attn_weight'])


        if np.mean(avg_attn_node) > 0.5:

        	average_attn[node] = {'average_attn':np.mean(avg_attn_node)}

        else:

        	average_attn[node] = {'average_attn':0.0}
        
    nx.set_node_attributes(g, average_attn)




##Remove edges by attention threshold
# vb_sc1 = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
# # Plot in form of degree
# deg_attn = {}
# for g in list_graphs:
# 	gp.remove_dummy_nodes(g)

# 	for edge in g.edges:

# 		if nx.get_edge_attributes(g,'attn_weight')[edge] < 0.5: # threshold attn_weights
# 			g.remove_edge(edge[0],edge[1])

# 	deg_attn = dict(g.degree)

# 	nx.set_node_attributes(g, deg_attn,'deg_attn')


# graph = list_graphs[graph_num]

# s_obj1, c_obj1, node_cb_obj1 = gv.show_graph(graph, nodes_coords, node_color_attribute='deg_attn',
# 											edge_color_attribute=None,c_map='inferno')



#vb_sc1.add_to_subplot(c_obj1)
# vb_sc1.add_to_subplot(s_obj1)
# visb_sc_shape1 = gv.get_visb_sc_shape(vb_sc1)
# vb_sc1.add_to_subplot(node_cb_obj1, row=visb_sc_shape1[0] - 1, col=visb_sc_shape1[0] + 0, width_max=300)

# vb_sc1.preview()


# Plot all the nodes on the template 
vb_sc2 = gv.visbrain_plot(template_mesh, caption='Visu on template mesh')
visb_sc_shape2 = gv.get_visb_sc_shape(vb_sc2)


# for i,g in enumerate(list_graphs):

# 	gp.remove_dummy_nodes(g)
# 	print('graph_num: ',i)

# 	#gp.sphere_nearest_neighbor_interpolation(g, template_mesh)
# 	nodes_coords = gp.graph_nodes_to_coords(g, 'Glasser2016_vertex_index', template_mesh)
# 	s_obj2, c_obj2, node_cb_obj2 = gv.show_graph(g, nodes_coords,node_color_attribute='average_attn', nodes_size=6, c_map='inferno')
# 	#vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
# 	vb_sc2.add_to_subplot(s_obj2)
# 	vb_sc2.add_to_subplot(node_cb_obj2, row=visb_sc_shape2[0] - 1, col=visb_sc_shape2[0] + 0, width_max=300)

# vb_sc2.preview()

attention_density_map = gv.attention_density_map(list_graphs, template_mesh, nb_iter=3, dt=0.5)

plt.figure()
plt.hist(attention_density_map, bins=50)
plt.show()

print(len(list_graphs))

visb_sc = gv.visbrain_plot(mesh=template_mesh, tex=attention_density_map,
                             caption='Template mesh',
                             cblabel='density',
                             cmap = 'hot')

visb_sc.preview()
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")

import os
import networkx as nx
import numpy as np
import tools.graph_processing as gp
import tools.plotly_extension as tp
import plotly.graph_objs as go
from os import listdir
from os.path import isfile, join

def compare_degree_real_simu(path_to_simu):
	
	degree_values = 20
	# real data
	path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs'

	# Get the meshes
	list_graphs = gp.load_graphs_in_list(path_to_graphs)
	degree_list = list()
	fig_labels = list()
	for ind, graph in enumerate(list_graphs):
		fig_labels.append('graph_'+str(ind))
		gp.remove_dummy_nodes(graph)
		print(len(graph.nodes))
		graph.remove_edges_from(nx.selfloop_edges(graph))
		degree_list.append(list(dict(nx.degree(graph)).values()))
	# compute the histos
	degree_histo = np.zeros((len(degree_list), degree_values))
	for i_d, dist in enumerate(degree_list):
		count = np.bincount(dist)
		for i,c in enumerate(count):
			degree_histo[i_d, i] += c
		degree_histo[i_d, :] = degree_histo[i_d, :]/np.sum(count)
	# lines for the plot
	x = list(range(degree_values))
	y = np.mean(degree_histo, 0)
	y_upper = y + np.std(degree_histo, 0)
	y_lower = y - np.std(degree_histo, 0)
	# error plot from real data
	fig_c = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='degree real data', color='rgb(20, 20, 200)')

	#simulated graphs
	
	path_to_graphs = path_to_simu  # path
	
		# Get the meshes
	list_graphs = gp.load_graphs_in_list(path_to_graphs)
	degree_list = list()
	fig_labels = list()
	for ind, graph in enumerate(list_graphs):
		fig_labels.append('simu_graph_'+str(ind))
		gp.remove_dummy_nodes(graph)
		print(len(graph.nodes))
		graph.remove_edges_from(nx.selfloop_edges(graph))
		degree_list.append(list(dict(nx.degree(graph)).values()))
	# compute the histos
	degree_histo = np.zeros((len(degree_list), degree_values))
	for i_d, dist in enumerate(degree_list):
		count = np.bincount(dist)
		for i,c in enumerate(count):
			degree_histo[i_d, i] += c
		degree_histo[i_d, :] = degree_histo[i_d, :]/np.sum(count)
	# lines for the plot
	y = np.mean(degree_histo, 0)
	y_upper = y + np.std(degree_histo, 0)
	y_lower = y - np.std(degree_histo, 0)
	# error plot from real data
	fig_c2 = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='degree simus', color='rgb(200, 20, 20)')
	fig_c.extend(fig_c2)
	fig = go.Figure(fig_c)



if __name__ == "__main__":


	fixed_path = './0/'

	directories = listdir(fixed_path)
	directories.sort()

	compare_degree_real_simu(fixed_path+directories[0]+"/graphs")
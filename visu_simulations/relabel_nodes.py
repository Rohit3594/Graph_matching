import os
import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching")
import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio
import numpy as np
import networkx as nx


if __name__ == "__main__":
	simus_run = 0
	path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/final_new_simu/0/noise_200,outliers_18/graphs/'
	list_graphs = gp.load_graphs_in_list(path_to_graphs)

	save_path = '/home/rohit/PhD_Work/GM_my_version/final_new_simu/0/Graph_relabeled/'

	counter = 0


	for g in list_graphs:

		sorted_G = nx.Graph() 
		sorted_G.add_nodes_from(sorted(g.nodes(data=True)))  # Sort the nodes of the graph by key
		sorted_G.add_edges_from(g.edges(data=True))
		
		nx.write_gpickle(sorted_G, os.path.join(save_path, "graphs","graph_" + str(counter) + ".gpickle"))

		print("sorted graph_",counter)

		counter+=1
import sys
#sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio
import numpy as np
import networkx as nx

if __name__ == "__main__":
	file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
	file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
	simus_run = 0
	#path_to_simu_graphs = '/home/rohit/PhD_Work/GM_my_version/final_new_simu/0/noise_1400,outliers_0/graphs'
	#path_to_real_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs'
	#gt = np.load('../data/simu_graph/0/test/0/noise_1400,outliers_8/ground_truth.npy')
	#path_to_simu_graphs = '../data/simu_graph/0/test/0/noise_1400,outliers_0/graphs'
	path_to_simu_graphs = '../data/simu_graph/simu_graph_rohit/noise_200,outliers_0/graphs'
	path_to_real_graphs = '../data/OASIS_full_batch/modified_graphs/'


	list_graphs_simu = gp.load_graphs_in_list(path_to_simu_graphs)
	list_graphs_real = 	gp.load_graphs_in_list(path_to_real_graphs)


	sphere_mesh = sio.load_mesh(file_sphere_mesh)
	mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))
	vb_sc1 = gv.visbrain_plot(mesh, caption='Visu of simulated graph on template mesh')


	mask_slice_coord = -15
	vb_sc = None
	inds_to_show = [0,2,9]
	graphs_to_show=[list_graphs_simu[i] for i in inds_to_show]
	for gr in graphs_to_show:#list_graphs_simu[6:12]:
		gp.sphere_nearest_neighbor_interpolation(gr, sphere_mesh)
		nodes_coords = gp.graph_nodes_to_coords(gr, 'ico100_7_vertex_index', mesh)
		nodes_mask = nodes_coords[:,2]>mask_slice_coord
		vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
		s_obj, c_obj, node_cb_obj = gv.show_graph(gr, nodes_coords,node_color_attribute=None, nodes_size=30, nodes_mask=nodes_mask, c_map='nipy_spectral')
		# vb_sc.add_to_subplot(s_obj)
		# vb_sc.add_to_subplot(c_obj)
		visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
		vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
		vb_sc.add_to_subplot(c_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
	vb_sc.preview()

	# # Stacking Graphs
	# concat_graphs = nx.disjoint_union(g,g1)
	# graph_stack = [g2,g3,g4]
	#
	# for graph in graph_stack:
	# 	concat_graphs = nx.disjoint_union(concat_graphs,graph)
	# gp.sphere_nearest_neighbor_interpolation(concat_graphs, sphere_mesh)
	#
	# #nodes_coords = gp.graph_nodes_to_coords(concat_graphs, 'ico100_7_vertex_index', sphere_mesh)
	# nodes_coords = gp.graph_nodes_to_coords(concat_graphs, 'ico100_7_vertex_index', mesh)


	# #vb_sc = gv.visbrain_plot(sphere_mesh, caption='Visu of simulated graph on template mesh')
	# vb_sc = gv.visbrain_plot(mesh, caption='Visu of simulated graph on template mesh')
	# s_obj, c_obj, node_cb_obj = gv.show_graph(concat_graphs, nodes_coords,node_color_attribute='label_gt', nodes_size=30)
	# vb_sc.add_to_subplot(s_obj)
	# vb_sc.add_to_subplot(c_obj)
	#
	# vb_sc.preview()
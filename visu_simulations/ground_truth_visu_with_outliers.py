import sys
sys.path.extend(['/home/rohit/PhD_data/Graph_matching'])
import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio
import numpy as np
import networkx as nx

if __name__ == "__main__":
	file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
	file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
	simus_run = 0
	path_to_graphs = '../data/simu_graph/0/test/0/noise_1400,outliers_8/graphs'
	list_graphs = gp.load_graphs_in_list(path_to_graphs)

	gt = np.load('../data/simu_graph/0/test/0/noise_1400,outliers_8/ground_truth.npy')

	gt_01 = gt[0][1]
	# gt_02 = gt[0][2]
	# gt_03 = gt[0][3]
	# gt_04 = gt[0][4]

	g = list_graphs[0]
	g1 = list_graphs[1]
	# g2 = list_graphs[2]
	# g3 = list_graphs[3]
	# g4 = list_graphs[4]

	dict_lab = {}
	for node in g1.nodes:
		if node >=20:
			dict_lab[node] = {'label':20}
		else:
			dict_lab[gt_01[node]] = {'label':node}
	nx.set_node_attributes(g1,dict_lab)



	# dict_lab = {}
	# for node in g2.nodes:
	# 	if node >=20:
	# 		dict_lab[node] = {'label':20}
	# 	else:
	# 		dict_lab[gt_02[node]] = {'label':node}
	# nx.set_node_attributes(g2,dict_lab)


	# dict_lab = {}
	# for node in g3.nodes:
	# 	if node >=20:
	# 		dict_lab[node] = {'label':20}
	# 	else:
	# 		dict_lab[gt_03[node]] = {'label':node}
	# nx.set_node_attributes(g3,dict_lab)



	# dict_lab = {}
	# for node in g4.nodes:
	# 	if node >=20:
	# 		dict_lab[node] = {'label':20}
	# 	else:
	# 		dict_lab[gt_04[node]] = {'label':node}
	# nx.set_node_attributes(g4,dict_lab)



	dict_lab = {}
	for node in g.nodes:
		if node >=20:
			dict_lab[node] = {'label':20}
		else:
			dict_lab[node] = {'label':node}
	nx.set_node_attributes(g,dict_lab)



	sphere_mesh = sio.load_mesh(file_sphere_mesh)
	mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

	gp.sphere_nearest_neighbor_interpolation(g, sphere_mesh)
	gp.sphere_nearest_neighbor_interpolation(g1, sphere_mesh)
	# gp.sphere_nearest_neighbor_interpolation(g2, sphere_mesh)
	# gp.sphere_nearest_neighbor_interpolation(g3, sphere_mesh)
	# gp.sphere_nearest_neighbor_interpolation(g4, sphere_mesh)


	nodes_coords = gp.graph_nodes_to_coords(g, 'ico100_7_vertex_index', sphere_mesh)
	nodes_coords_1 = gp.graph_nodes_to_coords(g1, 'ico100_7_vertex_index', sphere_mesh)
	# nodes_coords_2 = gp.graph_nodes_to_coords(g2, 'ico100_7_vertex_index', sphere_mesh)
	# nodes_coords_3 = gp.graph_nodes_to_coords(g3, 'ico100_7_vertex_index', sphere_mesh)
	# nodes_coords_4 = gp.graph_nodes_to_coords(g4, 'ico100_7_vertex_index', sphere_mesh)


	vb_sc = gv.visbrain_plot(sphere_mesh, caption='Visu of simulated graph on template mesh')
	s_obj, c_obj, node_cb_obj = gv.show_graph(g, nodes_coords,node_color_attribute='label')
	vb_sc.add_to_subplot(s_obj)
	#vb_sc.add_to_subplot(c_obj)
	
	s_obj, c_obj, node_cb_obj = gv.show_graph(g1, nodes_coords_1,node_color_attribute='label')
	#vb_sc.add_to_subplot(c_obj)
	vb_sc.add_to_subplot(s_obj)

	# s_obj, c_obj, node_cb_obj = gv.show_graph(g2, nodes_coords_2,node_color_attribute='label')
	# #vb_sc.add_to_subplot(c_obj)
	# vb_sc.add_to_subplot(s_obj)

	# s_obj, c_obj, node_cb_obj = gv.show_graph(g3, nodes_coords_3,node_color_attribute='label')
	# #vb_sc.add_to_subplot(c_obj)
	# vb_sc.add_to_subplot(s_obj)

	# s_obj, c_obj, node_cb_obj = gv.show_graph(g4, nodes_coords_4,node_color_attribute='label')
	# #vb_sc.add_to_subplot(c_obj)
	# vb_sc.add_to_subplot(s_obj)

	# show the plot on the screen
	vb_sc.preview()
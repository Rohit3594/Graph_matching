import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio
import numpy as np
import networkx as nx

def graph_remove_dummy_true(graph):
    nodes_dummy_true = [x for x,y in graph.nodes(data=True) if y['is_dummy']==True]
    graph.remove_nodes_from(nodes_dummy_true)
    print(len(graph.nodes))
    return graph



def dummy_for_visu(g,g1):
    
    gt_label = list(nx.get_node_attributes(g1,'label_gt').values())
    
    len_g1 = len(g1.nodes)

    labels = list(set(list(g.nodes)).symmetric_difference(set(gt_label)))

    if -1 in labels:
        labels.remove(-1)
        
    for i,j in enumerate(labels):
        
        g1.add_node(len_g1+i,label_gt=j,coord=[-0.0,-0.0,-0.0])





if __name__ == "__main__":
	file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
	file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
	simus_run = 0
	path_to_graphs = '../data/simu_graph/varied_outliers/0/test_graphs.gpickle'
	# list_graphs = gp.load_graphs_in_list(path_to_graphs)
	outliers_label = -1
	gt = np.load('../data/simu_graph/varied_outliers/0/new_groundtruth.npy',allow_pickle=True)

	gt_01 = gt[1]
	gt_02 = gt[2]
	gt_03 = gt[3]
	# gt_04 = gt[0][4]

	list_graphs = nx.read_gpickle(path_to_graphs)

	g  = list_graphs[0]
	g1 = list_graphs[1]
	g2 = list_graphs[2]
	g3 = list_graphs[3]


# IMportant: The graph with largest number of node should be used as a reference here for visualization 
	dict_lab = {}
	for n, node in enumerate(g.nodes):
		dict_lab[node] = {'label_gt':node}

	nx.set_node_attributes(g,dict_lab)


	dict_lab = {}
	for n, node in enumerate(g1.nodes):

		if node not in list(gt_01.values()):

			dict_lab[node] = {'label_gt':outliers_label}
		else:
			dict_lab[node] = {'label_gt':list(gt_01.keys())[list(gt_01.values()).index(node)]}


	nx.set_node_attributes(g1,dict_lab)
	print('gt_01: ',gt_01)
	print('g1 node attr len: ',nx.get_node_attributes(g1,'label_gt'))



	dict_lab = {}
	for n, node in enumerate(g2.nodes):
		if node not in list(gt_02.values()):

			dict_lab[node] = {'label_gt':outliers_label}
		else:
			dict_lab[node] = {'label_gt':list(gt_02.keys())[list(gt_02.values()).index(node)]}



	nx.set_node_attributes(g2,dict_lab)
	print('gt_02: ',gt_02)
	print('g2 node attrs: ',nx.get_node_attributes(g2,'label_gt'))



	dict_lab = {}
	for n, node in enumerate(g3.nodes):
		if node not in list(gt_03.values()):

			dict_lab[node] = {'label_gt':outliers_label}
		else:
			dict_lab[node] = {'label_gt':list(gt_03.keys())[list(gt_03.values()).index(node)]}
			


	nx.set_node_attributes(g3,dict_lab)
	print('gt_03: ',gt_03)
	print('g3 node attrs: ',nx.get_node_attributes(g3,'label_gt'))



	# nx.write_gpickle(g,'g_updated.gpickle')
	# nx.write_gpickle(g1,'g1_updated.gpickle')
	# nx.write_gpickle(g2,'g2_updated.gpickle')
	# nx.write_gpickle(g3,'g3_updated.gpickle')



# IMportant: The graph with largest number of node should be used as a reference here for visualization 

	dummy_for_visu(g,g1)
	dummy_for_visu(g,g2)
	dummy_for_visu(g,g3)



	sphere_mesh = sio.load_mesh(file_sphere_mesh)
	mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))

	g_simus= [g, g1, g2, g3]
	sphere_mesh = sio.load_mesh(file_sphere_mesh)

	vb_sc1 = gv.visbrain_plot(sphere_mesh, caption='Visu of simulated graph on template mesh')
	for grr in g_simus:
		gp.sphere_nearest_neighbor_interpolation(grr, sphere_mesh)
		nodes_coords = gp.graph_nodes_to_coords(grr, 'ico100_7_vertex_index', sphere_mesh)
		s_obj, c_obj, node_cb_obj = gv.show_graph(grr, nodes_coords,node_color_attribute='label_gt', nodes_size=30, c_map='nipy_spectral')
		#vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
		vb_sc1.add_to_subplot(s_obj)

	# Stacking Graphs
	# concat_graphs = nx.disjoint_union(g,g1)
	# graph_stack = [g2,g3]


	# for graph in graph_stack:
	# 	concat_graphs = nx.disjoint_union(concat_graphs,graph)

	# print(list(nx.get_node_attributes(concat_graphs,'label_gt')))


	# gp.sphere_nearest_neighbor_interpolation(concat_graphs, sphere_mesh)


	# nodes_coords = gp.graph_nodes_to_coords(concat_graphs, 'ico100_7_vertex_index', mesh)

	# print('LEN NODES COORDS: ', len(nodes_coords))
	# print('LEN CONCAT NODES', len(concat_graphs.nodes))


	# s_obj, c_obj, node_cb_obj = gv.show_graph(concat_graphs, nodes_coords,node_color_attribute='label_gt', nodes_size=30, c_map='nipy_spectral')
	# visb_sc_shape = gv.get_visb_sc_shape(vb_sc1)
	# #vb_sc1.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
	# vb_sc1.add_to_subplot(s_obj)

	vb_sc1.preview()




	mask_slice_coord = -150
	vb_sc = None
	for gr in g_simus:
		#gr = nx.relabel.convert_node_labels_to_integers(gr)
		# sorted_G = nx.Graph() 
		# sorted_G.add_nodes_from(sorted(gr.nodes(data=True)))  # Sort the nodes of the graph by key
		# sorted_G.add_edges_from(gr.edges(data=True))

		gp.sphere_nearest_neighbor_interpolation(gr, sphere_mesh)
		nodes_coords = gp.graph_nodes_to_coords(gr, 'ico100_7_vertex_index', mesh)
		nodes_mask = nodes_coords[:,2]>mask_slice_coord
		vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
		s_obj, c_obj, node_cb_obj = gv.show_graph(gr, nodes_coords,node_color_attribute='label_gt', nodes_size=30, nodes_mask=nodes_mask, c_map='nipy_spectral')
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
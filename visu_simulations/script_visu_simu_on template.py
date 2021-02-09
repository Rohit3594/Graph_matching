import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio

if __name__ == "__main__":
    file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
    simus_run = 0
    path_to_graphs = '../data/simu_graph/noise_0,outliers_0/'+str(simus_run)+'/graphs'
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    print(len(list_graphs))
    # Get the meshes
    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))
    graph = list_graphs[0]
    gp.remove_dummy_nodes(graph)
    gp.sphere_nearest_neighbor_interpolation(graph, sphere_mesh)
    nodes_coords = gp.graph_nodes_to_coords(graph, 'ico100_7_vertex_index', mesh)
    vb_sc = gv.visbrain_plot(mesh, caption='Visu of simulated graph on template mesh')
    s_obj, c_obj, node_cb_obj = gv.show_graph(graph, nodes_coords)
    vb_sc.add_to_subplot(c_obj)
    vb_sc.add_to_subplot(s_obj)

    # show the plot on the screen
    vb_sc.preview()


import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp


if __name__ == "__main__":

    path_to_graphs = '../data/OASIS_full_batch/'
    #path_to_graphs = '../data/simu_graph/' #simulated graphs

    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)

    # using an average mesh as template
    template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))


    density_map = gv.nodes_density_map(list_graphs, mesh, nb_iter=5, dt=0.5)

    visb_sc = gv.visbrain_plot(mesh=mesh,
                            tex=density_map,
                            caption='density map',
                            cmap="jet")


    visb_sc.preview()






    # ## visualization of nodes as spheres
    # vb_sc = gv.visbrain_plot(mesh)
    # for g in list_graphs:
    #     s_obj, c_obj = gv.show_graph(g, mesh, 'red', edge_attribute=None)
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()
    #
    # # using a sphere mesh as template
    # template_mesh = '/mnt/data/work/python_sandBox/Graph_matching/data/template_mesh/ico100_7.gii'
    # mesh = sio.load_mesh(template_mesh)
    #
    # vb_sc = gv.visbrain_plot(mesh)
    # for g in list_graphs:
    #     s_obj, c_obj = gv.show_graph(g, mesh, 'red', edge_attribute=None)
    #     vb_sc.add_to_subplot(s_obj)
    #
    # vb_sc.preview()

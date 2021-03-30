import sys
sys.path.extend(['/home/rohit/PhD_Work/GM_my_version/Graph_matching'])
import slam.io as sio
import tools.graph_visu as gv
import tools.graph_processing as gp


if __name__ == "__main__":
    template_mesh = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    path_to_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_full_batch/modified_graphs/'
    list_graphs = gp.load_graphs_in_list(path_to_graphs)

    # Get the mesh
    mesh = gv.reg_mesh(sio.load_mesh(template_mesh))

    vb_sc = None
    for g in list_graphs[:4]:
        #s_obj, c_obj = show_graph(g, mesh, 'red', edge_attribute='geodesic_distance', mask_slice_coord=-15)
        s_obj, c_obj = gv.show_graph(g, mesh, 'red')
        vb_sc = gv.visbrain_plot(mesh, visb_sc=vb_sc)
        visb_sc_shape = gv.get_visb_sc_shape(vb_sc)
        vb_sc.add_to_subplot(s_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)
        vb_sc.add_to_subplot(c_obj, row=visb_sc_shape[0] - 1, col=visb_sc_shape[1]- 1)

    # s_obj, c_obj = show_graph(list_graphs[0], mesh, edge_attribute='geodesic_distance')
    # vb_sc = visbrain_plot(mesh)
    # vb_sc.add_to_subplot(s_obj)
    # vb_sc.add_to_subplot(c_obj)
    vb_sc.preview()

import slam.io as sio
import networkx as nx
import tools.graph_visu as gv
import tools.graph_processing as gp


if __name__ == "__main__":
    #"/home/rohit/PhD_Work/GM_my_version/Graph_matching/data"
    file_mesh = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/example_individual_OASIS_0061/rh.white.gii'
    file_basins = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/example_individual_OASIS_0061/alpha0.03_an0_dn20_r1.5_R_area50FilteredTexture.gii'
    file_graph = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/example_individual_OASIS_0061/OAS1_0061_rh_pitgraph.gpickle'

    graph = nx.read_gpickle(file_graph)
    # Get the mesh
    mesh = gv.reg_mesh(sio.load_mesh(file_mesh))
    import trimesh.smoothing as tms
    mesh = tms.filter_laplacian(mesh,iterations=80)
    #mesh.show()
    tex_basins = sio.load_texture(file_basins)
    vb_sc = gv.visbrain_plot(mesh,tex=tex_basins.darray[2], cmap='tab20c' )
    s_obj, c_obj = gv.show_graph(graph, mesh, 'red', edge_attribute=None, mask_slice_coord=-15, transl=[0,0,2])
    vb_sc.add_to_subplot(c_obj)
    vb_sc.add_to_subplot(s_obj)


    vb_sc.preview()

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio

if __name__ == "__main__":
    file_template_mesh = '../data/template_mesh/lh.OASIS_testGrp_average_inflated.gii'
    file_sphere_mesh = '../data/template_mesh/ico100_7.gii'
    #simus_run = 0
    #path_to_graphs = '../data/simu_graph/noise_70,outliers_0/'+str(simus_run)+'/graphs'
    path_to_graphs = '../data/simu_graph/0/noise_1000,outliers_0/graphs/'
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    # path_ref_graph = '../data/simu_graph/noise_0,outliers_0/'+str(simus_run)+'/ground_truth.gpickle'
    # graph_ref = nx.read_gpickle(path_ref_graph)
    # list_graphs.append(graph_ref)

    print(len(list_graphs))
    # Get the meshes
    sphere_mesh = sio.load_mesh(file_sphere_mesh)
    mesh = gv.reg_mesh(sio.load_mesh(file_template_mesh))
    for graph in list_graphs:
        gp.remove_dummy_nodes(graph)
        print(len(graph.nodes))
        gp.sphere_nearest_neighbor_interpolation(graph, sphere_mesh)

    density_map = gv.nodes_density_map(list_graphs, mesh, nb_iter=3, dt=0.5)

    plt.figure()
    plt.hist(density_map, bins=50)
    plt.show()

    visb_sc = gv.visbrain_plot(mesh=mesh,
                            tex=density_map,
                            caption='density map',
                            cmap="jet",
                            clim=(0, 0.03)) #clim = cmap range, default = (min(data), max(data))



    visb_sc.preview()
    #
    # visb_sc2 = gv.visbrain_plot(mesh=sphere_mesh,
    #                         tex=density_map,
    #                         caption='density map',
    #                         cmap="jet")
    #
    #
    # visb_sc2.preview()

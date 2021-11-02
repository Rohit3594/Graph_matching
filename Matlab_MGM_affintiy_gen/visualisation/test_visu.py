import numpy as np
import networkx as nx
import slam.topology as stop
import slam.plot as splt
import slam.generate_parametric_surfaces as sps
import slam.mapping as smap
import numpy as np
from visbrain.objects import VispyObj, SourceObj

my_graph = nx.read_gpickle("../generation_graphes/generated_graphs_big/noise_3.0,outliers_4/0/ref_graph.gpickle")
my_graph_noisy = nx.read_gpickle("../generation_graphes/generated_graphs_big/noise_3.0,outliers_4/0/noisy_graph.gpickle")

node_array_ref = np.array([my_graph.nodes[my_node]["coord"] for my_node in my_graph.nodes])
node_array_noisy = np.array([my_graph_noisy.nodes[my_node]["coord"] for my_node in my_graph_noisy.nodes])


sphere_mesh = sps.generate_sphere_random_sampling(1000, 100)
visb_sc = splt.visbrain_plot(mesh=sphere_mesh)

#s_ref = SourceObj('ref', node_array_ref, color='red', symbol='square',
                          #radius_min=10, alpha=0.2)
#s_noisy = SourceObj('noisy', node_array_noisy, color='blue', symbol='square',
                          #radius_min=10, alpha=0.2)

s_north = SourceObj("north", np.array([[0,0,100]]), color="red", symbol="square", radius_min=10, alpha=0.2)

#visb_sc.add_to_subplot(s_ref)
#visb_sc.add_to_subplot(s_noisy)
visb_sc.add_to_subplot(s_north)
visb_sc.preview()

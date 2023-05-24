import os
import networkx as nx
import numpy as np
import tools.graph_processing as gp
import tools.plotly_extension as tp
import plotly.graph_objs as go

def mean_depth(G):
    all_depth = [depth for node,depth in list(G.nodes.data('depth'))]
    #mean_geo = np.array(all_geo).mean()
    #std = np.std(all_geo)
    
    return all_depth


def graph_remove_dummy_nodes(graph):
    nodes_dummy_true = [x for x,y in graph.nodes(data=True) if y['is_dummy']==True]
    graph.remove_nodes_from(nodes_dummy_true)
    #print(len(graph.nodes))
    return graph



if __name__ == "__main__":
    max_depth_values = 5
    nb_bins = 20
    # real data
    path_to_graphs =  './data/Oasis_original_new_with_dummy/modified_graphs/'

    # Get the meshes
    list_graphs = [nx.read_gpickle(path_to_graphs+'/'+graph) for graph in np.sort(os.listdir(path_to_graphs))]
    list_graphs = [graph_remove_dummy_nodes(g) for g in list_graphs]


    depth_list = list()
    fig_labels = list()

    for ind, graph in enumerate(list_graphs):
        fig_labels.append('graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        print('nb_nodes:',len(graph.nodes))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        depth_list.append(mean_depth(graph))
    # compute the histos

    depth_list_flat = [item for sublist in depth_list for item in sublist] #flatten list to one single list

    print('minimum depth: ',min(depth_list_flat))
    print('maximum depth: ',max(depth_list_flat))

    bins = np.arange(min(depth_list_flat),max(depth_list_flat), max(depth_list_flat)/nb_bins)

    depth_histo = list()
    for i_d, dist in enumerate(depth_list):
        hist, bin_edges = np.histogram(dist, bins, density=True)
        depth_histo.append(hist)

    depth_histo = np.array(depth_histo)
    # lines for the plot
    x = np.linspace(min(depth_list_flat),max(depth_list_flat),100)

    y = np.mean(depth_histo, 0)
    print('y', y)
    y_upper = y + np.std(depth_histo, 0)
    y_lower = y - np.std(depth_histo, 0)
    # error plot from real data
    fig_c = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='depth real data', color='rgb(20, 20, 200)')

    fig = go.Figure(fig_c)

    fig.update_layout(
        xaxis_title='Depth',
        yaxis_title='Proportion',
        title= 'Depth distribution',
        hovermode="x",
        font=dict(
        size=30,
    )
    )
    #fig.show(renderer="browser")
    fig.write_html('first_figure.html', auto_open=True)
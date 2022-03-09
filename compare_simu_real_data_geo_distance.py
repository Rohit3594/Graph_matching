import os
import networkx as nx
import numpy as np
import tools.graph_processing as gp
import tools.plotly_extension as tp
import plotly.graph_objs as go

def mean_edge_len(G):
    
    all_geo = [z['geodesic_distance'] for x,y,z in list(G.edges.data())]
    mean_geo = np.array(all_geo).mean()
    std = np.std(all_geo)
    
    return all_geo


if __name__ == "__main__":
    geo_values = 200
    # real data
    path_to_graphs = './data/OASIS_full_batch/modified_graphs'

    # Get the meshes
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    geo_list = list()
    fig_labels = list()

    for ind, graph in enumerate(list_graphs):
        fig_labels.append('graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        print(len(graph.nodes))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        geo_list.append(mean_edge_len(graph))
    # compute the histos
    geo_histo = np.zeros((len(geo_list), geo_values))
    for i_d, dist in enumerate(geo_list):
        count = np.bincount(dist)
        print("len bincount",len(count))
        for i,c in enumerate(count):
            print(i,c)
            geo_histo[i_d, i] += c
        geo_histo[i_d, :] = geo_histo[i_d, :]/np.sum(count)
    # lines for the plot
    x = list(range(geo_values))
    y = np.mean(geo_histo, 0)
    y_upper = y + np.std(geo_histo, 0)
    y_lower = y - np.std(geo_histo, 0)
    # error plot from real data
    fig_c = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='geo real data', color='rgb(20, 20, 200)')

    #simulated graphs
    path_to_graphs = './data/simu_graph/simu_test/0/noise_100,outliers_20/graphs/'
        # Get the meshes
    list_graphs = gp.load_graphs_in_order(path_to_graphs)

    geo_list = list()
    fig_labels = list()
    for ind, graph in enumerate(list_graphs):
        fig_labels.append('simu_graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        print(len(graph.nodes))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        geo_list.append(mean_edge_len(graph))
    # compute the histos
    geo_histo = np.zeros((len(geo_list), geo_values))
    for i_d, dist in enumerate(geo_list):
        count = np.bincount(dist)
        for i,c in enumerate(count):
            geo_histo[i_d, i] += c
        geo_histo[i_d, :] = geo_histo[i_d, :]/np.sum(count)
    # lines for the plot
    y = np.mean(geo_histo, 0)
    y_upper = y + np.std(geo_histo, 0)
    y_lower = y - np.std(geo_histo, 0)
    # error plot from real data
    fig_c2 = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='geo simus', color='rgb(200, 20, 20)')
    fig_c.extend(fig_c2)
    fig = go.Figure(fig_c)

        # fig.add_trace(go.Scatter(
        #     name=fig_labels[i_d],
        #     x=x,
        #     y=degree_histo[i_d, :],
        #     mode='lines',
        #     line=dict(color='rgb(25, 25, 180)')))

    fig.update_layout(
        yaxis_title='proportion',
        title='distribution of geodesic_distance noise_100,outliers_varied',
        hovermode="x"
    )
    #fig.show(renderer="browser")
    fig.write_html('first_figure.html', auto_open=True)


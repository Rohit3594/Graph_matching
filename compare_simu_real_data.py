import os
import networkx as nx
import numpy as np
import tools.graph_processing as gp
import tools.plotly_extension as tp
import plotly.graph_objs as go


if __name__ == "__main__":
    degree_values = 20
    # real data
    path_to_graphs = 'data/OASIS_full_batch/modified_graphs'

    # Get the meshes
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    degree_list = list()
    fig_labels = list()
    for ind, graph in enumerate(list_graphs):
        fig_labels.append('graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        print(len(graph.nodes))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        degree_list.append(list(dict(nx.degree(graph)).values()))
    # compute the histos
    degree_histo = np.zeros((len(degree_list), degree_values))
    for i_d, dist in enumerate(degree_list):
        count = np.bincount(dist)
        for i,c in enumerate(count):
            degree_histo[i_d, i] += c
        degree_histo[i_d, :] = degree_histo[i_d, :]/np.sum(count)
    # lines for the plot
    x = list(range(degree_values))
    y = np.mean(degree_histo, 0)
    y_upper = y + np.std(degree_histo, 0)
    y_lower = y - np.std(degree_histo, 0)
    # error plot from real data
    fig_c = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='degree real data', color='rgb(20, 20, 200)')

    #simulated graphs
    path_to_graphs = 'data/simu_graph/simu_graph_rohit/0/noise_1000,outliers_0/graphs/'
        # Get the meshes
    list_graphs = gp.load_graphs_in_list(path_to_graphs)
    degree_list = list()
    fig_labels = list()
    for ind, graph in enumerate(list_graphs):
        fig_labels.append('simu_graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        print(len(graph.nodes))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        degree_list.append(list(dict(nx.degree(graph)).values()))
    # compute the histos
    degree_histo = np.zeros((len(degree_list), degree_values))
    for i_d, dist in enumerate(degree_list):
        count = np.bincount(dist)
        for i,c in enumerate(count):
            degree_histo[i_d, i] += c
        degree_histo[i_d, :] = degree_histo[i_d, :]/np.sum(count)
    # lines for the plot
    y = np.mean(degree_histo, 0)
    y_upper = y + np.std(degree_histo, 0)
    y_lower = y - np.std(degree_histo, 0)
    # error plot from real data
    fig_c2 = tp.error_plot(x=x, y=y, y_lower=y_lower, y_upper=y_upper, line_label='degree simus', color='rgb(200, 20, 20)')
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
        title='distribution of degree',
        hovermode="x"
    )
    #fig.show(renderer="browser")
    fig.write_html('first_figure.html', auto_open=True)


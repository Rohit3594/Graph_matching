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


def graph_remove_dummy_nodes(graph):
    nodes_dummy_true = [x for x,y in graph.nodes(data=True) if y['is_dummy']==True]
    graph.remove_nodes_from(nodes_dummy_true)
    #print(len(graph.nodes))
    return graph



if __name__ == "__main__":
    geo_values = 200
    # real data
    path_to_graphs =  './data/Oasis_original_new_with_dummy/modified_graphs/'

    # Get the meshes
    #list_graphs = gp.load_graphs_in_list(path_to_graphs)
    list_graphs = [nx.read_gpickle(path_to_graphs+'/'+graph) for graph in np.sort(os.listdir(path_to_graphs))]
    list_graphs = [graph_remove_dummy_nodes(g) for g in list_graphs]

    geo_list = list()
    fig_labels = list()

    for ind, graph in enumerate(list_graphs):
        fig_labels.append('graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        print('nb_nodes:',len(graph.nodes))
        graph.remove_edges_from(nx.selfloop_edges(graph))
        geo_list.append(mean_edge_len(graph))
    # compute the histos
    geo_histo = np.zeros((len(geo_list), geo_values))
    for i_d, dist in enumerate(geo_list):
        count = np.bincount(dist)
        #print("len bincount",len(count))
        for i,c in enumerate(count):
            #print(i,c)
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
    path_to_graphs = './data/simu_graph/NEW_SIMUS_JULY_11/3/noise_1000,outliers_varied/graphs/'
        # Get the meshes
    #list_graphs = gp.load_graphs_in_order(path_to_graphs)
    list_graphs = [nx.read_gpickle(path_to_graphs+'/'+graph) for graph in np.sort(os.listdir(path_to_graphs))]

    geo_list = list()
    fig_labels = list()
    for ind, graph in enumerate(list_graphs):
        fig_labels.append('simu_graph_'+str(ind))
        gp.remove_dummy_nodes(graph)
        #print('nb_nodes:',len(graph.nodes))
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
        yaxis_title='Proportion',
        xaxis_title='Geodesic distance',
        title='distribution of geodesic_distance noise_1000,outliers_varied',
        hovermode="x",
        font=dict(
        size=30,
    )

    )
    #fig.show(renderer="browser")
    fig.write_html('first_figure.html', auto_open=True)


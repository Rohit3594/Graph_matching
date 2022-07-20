import tools.graph_processing as gp
import numpy as np
import networkx as nx
import pickle as p
from visbrain.objects import SourceObj, ColorbarObj


def label_nodes_according_to_coord(graph_no_dummy, template_mesh, coord_dim=1):
    nodes_coords = gp.graph_nodes_to_coords(graph_no_dummy, 'ico100_7_vertex_index', template_mesh)

    one_nodes_coords = nodes_coords[:, coord_dim]
    one_nodes_coords_scaled = (one_nodes_coords - np.min(one_nodes_coords))/(np.max(one_nodes_coords)-np.min(one_nodes_coords))

    #one_nodes_coords_scaled = np.random.rand(len(nodes_coords))
    # initialise the dict for atttributes
    nodes_attributes = {}
    # Fill the dictionnary with the nd_array attribute
    for ind, node in enumerate(graph_no_dummy.nodes):
        nodes_attributes[node] = {"label_color": one_nodes_coords_scaled[ind]}

    nx.set_node_attributes(graph_no_dummy, nodes_attributes)
    return one_nodes_coords_scaled


def get_clusters_from_assignment(list_graphs, matching_matrix, largest_ind, mesh, labelling_attribute_name):
    nb_graphs = len(list_graphs)
    g_l = list_graphs[largest_ind]
    color_label_ordered = label_nodes_according_to_coord(g_l, mesh, coord_dim=1)
    r_perm = p.load(open("../data/r_perm.gpickle","rb"))
    color_label = color_label_ordered[r_perm]
    gp.add_nodes_attribute(g_l, color_label, labelling_attribute_name)
    default_value = -0.1#0.05
    nb_nodes = len(g_l.nodes)
    row_scope = range(largest_ind * nb_nodes, (largest_ind + 1) * nb_nodes)

    #nb_unmatched = 0
    for i in range(nb_graphs):
        col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
        perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
        transfered_labels = np.ones(101)*default_value
        for node_indx,ind in enumerate(row_scope):
            match_index = np.where(perm_X[node_indx,:]==1)[0]

            if len(match_index)>0:
                transfered_labels[match_index[0]] = color_label[node_indx]
        #nb_unmatched += np.sum(transfered_labels==default_value)
        g = list_graphs[i]
        gp.add_nodes_attribute(g, transfered_labels, labelling_attribute_name)
    return transfered_labels


def get_clusters_from_assignment_hippi(list_graphs, matching_matrix, largest_ind, mesh, labelling_attribute_name):
    nb_graphs = len(list_graphs)
    g_l = list_graphs[largest_ind]
    color_label_ordered = label_nodes_according_to_coord(g_l, mesh, coord_dim=1)
    r_perm = p.load(open("/mnt/data/work/python_sandBox/Graph_matching/data/r_perm.gpickle","rb"))
    color_label = color_label_ordered[r_perm]
    gp.add_nodes_attribute(g_l, color_label, labelling_attribute_name)
    default_value = -0.1#0.05
    nb_nodes = len(g_l.nodes)

    for j in range(len(list_graphs)):

        grph = list_graphs[j]
        nodes_to_remove = gp.remove_dummy_nodes(grph)
        nodes_to_remove = np.where(np.array(nodes_to_remove)==False)

        grph.remove_nodes_from(list(nodes_to_remove[0]))
        nb_nodes = len(grph.nodes)
        row_scope = range(j * nb_nodes, (j + 1) * nb_nodes)

        print(len(grph.nodes))

        if len(grph.nodes)==101:
            break


    print(matching_matrix.shape)
    last_index = 0

    nb_unmatched = 0
    for i in range(nb_graphs):

        #g = list_graphs[i]
        #g=p.load(open("../data/OASIS_full_batch/modified_graphs/graph_"+str(i)+".gpickle","rb"))
        g = list_graphs[i]
        nodes_to_remove = gp.remove_dummy_nodes(g)
        nodes_to_remove = np.where(np.array(nodes_to_remove)==False)
        g.remove_nodes_from(list(nodes_to_remove[0]))
        nb_nodes = len(g.nodes)

        print(len(g.nodes))

        if i == 0:
            col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)
            prev_nb_nodes = nb_nodes
            perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
            transfered_labels = np.ones(nb_nodes)*default_value
            last_index+=nb_nodes
        else:
            col_scope = range(last_index, last_index+nb_nodes)
            last_index += nb_nodes
            perm_X = np.array(matching_matrix[np.ix_(row_scope, col_scope)], dtype=int) #Iterate through each Perm Matrix X fixing the largest graph
            transfered_labels = np.ones(nb_nodes)*default_value


        print(col_scope)

        #nb_nodes = len(g.nodes)
        #col_scope = range(i * nb_nodes, (i + 1) * nb_nodes)

        for node_indx,ind in enumerate(row_scope):
            match_index = np.where(perm_X[node_indx,:]==1)[0]

            if len(match_index)>0:
                transfered_labels[match_index[0]] = color_label[node_indx]
        gp.add_nodes_attribute(g, transfered_labels, labelling_attribute_name)
    return transfered_labels


def create_clusters_lists(list_graphs, label_attribute="label_dbscan"):
    """
    Given a list of graphs, return a list of list that represents the clusters.
    Each inside list represent one cluster and each elemnt of the cluster is
    a tuple (graph_number, node_in_the_graph).
    """

    result_dict = {}

    for i_graph, graph in enumerate(list_graphs):
        for node in graph.nodes:

            label_cluster = graph.nodes[node][label_attribute]
            if label_cluster != -1 and not graph.nodes[node]["is_dummy"]:
                if label_cluster in result_dict:
                    result_dict[label_cluster].append((i_graph, node))
                else:
                    result_dict[label_cluster] = [(i_graph, node)]

    # We make sure that every clusters have more than one element
    return {i: result_dict[i] for i in result_dict if len(result_dict[i]) > 1}


def get_centroid_clusters(list_graphs, clusters_dict):
    """
    Return a dictionary which gives for each cluster the belonging point
    which is the closest to the centroid
    """

    result_dict = {}

    for cluster_key in clusters_dict:

        # initialise the matrix which holds the position of all the point in the cluster
        position_mat = np.zeros((len(clusters_dict[cluster_key]), 3))

        # fill the matrix
        for elem_i, (graph_num, node) in enumerate(clusters_dict[cluster_key]):
            graph = list_graphs[graph_num]
            position_mat[elem_i, :] = graph.nodes[node]["sphere_3dcoords"]

        # get the centroid
        centroid = position_mat.mean(0)

        # get the closest point to the centroid
        min_distance = -1
        for graph_num, node in clusters_dict[cluster_key]:

            graph = list_graphs[graph_num]
            position_node = graph.nodes[node]["sphere_3dcoords"]
            distance_to_centroid = np.linalg.norm(centroid - position_node)

            if distance_to_centroid < min_distance or min_distance == -1:
                min_distance = distance_to_centroid
                point_to_save = (graph_num, node)

        result_dict[cluster_key] = point_to_save

    return result_dict


def get_all_silhouette_value(list_graphs, cluster_dict):
    """
    Return a dict with all the silhouette value that can be calculated for each cluster
    """

    result_dict = {}

    for cluster_key in cluster_dict:
        for main_counter in range(len(cluster_dict[cluster_key])):

            graph_main, node_main = cluster_dict[cluster_key][main_counter]
            vector_1 = list_graphs[graph_main].nodes[node_main]["sphere_3dcoords"]

            # We calculate the inter cluster similarity
            a_list = []
            for inter_cluster_counter in range(len(cluster_dict[cluster_key])):

                if main_counter != inter_cluster_counter:
                    graph_inter, node_inter = cluster_dict[cluster_key][inter_cluster_counter]
                    vector_2 = list_graphs[graph_inter].nodes[node_inter]["sphere_3dcoords"]

                    distance = np.linalg.norm(vector_1 - vector_2)
                    a_list.append(distance)

            a = np.mean(a_list)

            # We calculate the average distance with points from other cluster
            b_list = []
            for cluster_intra_key in cluster_dict:

                if cluster_intra_key != cluster_key:

                    distance_list = []
                    for intra_cluster_counter in range(len(cluster_dict[cluster_intra_key])):
                        graph_intra, node_intra = cluster_dict[cluster_intra_key][intra_cluster_counter]
                        vector_2 = list_graphs[graph_intra].nodes[node_intra]["sphere_3dcoords"]

                        distance = np.linalg.norm(vector_1 - vector_2)
                        distance_list.append(distance)

                    b_list.append(np.mean(distance_list))

            b = np.min(b_list)

            silouhette = (b - a) / max(b, a)

            if cluster_key in result_dict:
                result_dict[cluster_key].append(silouhette)
            else:
                result_dict[cluster_key] = [silouhette]

    return result_dict


def get_silhouette_per_cluster(silhouette_dict):
    nb_clusters = len(silhouette_dict)
    silhouette_data = np.zeros(nb_clusters)
    cluster_nb_nodes = np.zeros(nb_clusters)

    # Get the data
    for cluster_i, cluster_key in enumerate(silhouette_dict):
        silhouette_data[cluster_i] = np.mean(silhouette_dict[cluster_key])
        cluster_nb_nodes[cluster_i] = len(silhouette_dict[cluster_key])
    return silhouette_data, cluster_nb_nodes


def get_silhouette_source_obj(centroid_dict, list_graphs, silhouette_data, mesh, c_map='jet', clim=None):
    """
    Return a SourceObj that represent the silhouette value at each cluster centroid
    """

    nb_clusters = len(list(centroid_dict.keys()))
    silhouette_3Dpos = np.zeros((nb_clusters, 3))

    # Get the data
    for cluster_i, cluster_key in enumerate(centroid_dict):
        graph_num, node = centroid_dict[cluster_key]
        graph = list_graphs[graph_num]

        vertex = graph.nodes[node]["ico100_7_vertex_index"]
        vertex_pos = mesh.vertices[vertex, :]
        # print(vertex_pos)
        # print(silhouette_3Dpos.shape)
        silhouette_3Dpos[cluster_i, :] = vertex_pos

    # Create the source obj
    transl_bary = np.mean(silhouette_3Dpos)
    nodes_coords = 1.01*(silhouette_3Dpos-transl_bary)+transl_bary
    silhouette_text = [str(elem) for elem in silhouette_data]
    s_obj = SourceObj('nodes', nodes_coords, color='black',
                        edge_color='black', symbol='o', edge_width=2.,
                        radius_min=60., radius_max=60., alpha=.9)
    s_obj.color_sources(data=silhouette_data, cmap=c_map,clim=clim)
    # Get the colorbar of the source object
    CBAR_STATE = dict(cbtxtsz=30, txtsz=30., width=.1, cbtxtsh=3.,
                  rect=(-.3, -2., 1., 4.), txtcolor='k', border=False)
    cb_obj = ColorbarObj(s_obj, **CBAR_STATE)

    # source_silhouette = SourceObj("source_silhouette",
    #                               nodes_coords,
    #                               silhouette_data,
    #                               text=silhouette_text,
    #                               radius_min=20,
    #                               radius_max=30,
    #                               text_size=50)

    return s_obj, cb_obj


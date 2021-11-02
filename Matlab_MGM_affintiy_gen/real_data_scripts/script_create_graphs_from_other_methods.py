import numpy as np
import slam.io as sio
import networkx as nx
import os
import argparse


def create_graph_from_texture(texture, mesh_sphere, method="media"):
    """
    Based on a texture representing the sulcal pits, create a nx graph
    to be used by visualisation script already available.
    Textures can come from either the media method or the neuroimage method.
    """

    # initialise the graph
    graph = nx.Graph()

    # initialise the list of tuple (node, attributes)
    list_nodes = []

    # Go through the texture and add the pits
    counter_node = 0
    for i_texture, label in enumerate(texture):

        # Neuroimage method
        if method=="neuroimage":
            if label != 0: # If the node is a pit

                # Initialise the dict that defines the attribute of the node
                attributes_node = {"is_dummy": False}

                # Add the label
                attributes_node["label_dbscan"] = label

                # add the position on the sphere
                pos_sphere = mesh_sphere.vertices[i_texture,:]
                attributes_node["sphere_3dcoords"] = pos_sphere

                # add the vertex index
                attributes_node["ico100_7_vertex_index"] = i_texture

                # Add the node to the list
                list_nodes.append((counter_node, attributes_node))

                counter_node += 1


        # Media method
        elif method=="media":
            if label != 0:
                
                # Initialise the dict that defines the attribute of the node
                attributes_node = {"is_dummy": False}

                # Add the label
                if label == -2:
                    attributes_node["label_dbscan"] = -1
                else:
                    attributes_node["label_dbscan"] = label

                # add the position on the sphere
                pos_sphere = mesh_sphere.vertices[i_texture,:]
                attributes_node["sphere_3dcoords"] = pos_sphere

                # add the vertex index
                attributes_node["ico100_7_vertex_index"] = i_texture

                # Add the node to the list
                list_nodes.append((counter_node, attributes_node))

                counter_node += 1

        else:
            print("Wrong method name")

    # Add the nodes to the graph
    graph.add_nodes_from(list_nodes)

    return graph


def create_modified_graphs(path_to_read, path_mesh, method="media"):
    """
    Based on a folder filled with texture representing sulcal pits,
    create a folder modified_graphs and fill it with the corresponding
    sulcal pits graphs
    """

    # load the sphere mesh
    mesh_sphere = sio.load_mesh(path_mesh)

    # load the texture in the folder
    list_texture = []
    for texture_name in os.listdir(path_to_read):

        texture_path = os.path.join(path_to_read, texture_name)
        if (not os.path.isdir(texture_path)) and texture_name[0] != ".":
            texture = sio.load_texture(texture_path).darray[0,:]
            list_texture.append(texture)

    # Create the folder for the graphs
    graph_folder_path = os.path.join(path_to_read, "modified_graphs")
    if not os.path.isdir(graph_folder_path):
        os.mkdir(graph_folder_path)

    # Go through every texture and create the appropriate graph
    for i_texture, texture in enumerate(list_texture):

        graph = create_graph_from_texture(texture, mesh_sphere, method)

        # save the graph in the appropriate folder
        graph_path = os.path.join(graph_folder_path, "graph_"+str(i_texture)+".gpickle")
        nx.write_gpickle(graph, graph_path)


        
if __name__ == "__main__":

    # We parse the argument from command line
    parser = argparse.ArgumentParser(description="From a set folder of texture of sulcal pits, create the graph equivalent to be used in visualisation scripts")
    parser.add_argument("path_to_folder", help="path to a folder which contains the textures")
    parser.add_argument("--method",
                        help="the method used to generate the textures",
                        default="media")
    parser.add_argument("--path_mesh",
                        help="path to the mesh file",
                        default="/Users/Nathan/stageINT/stage_nathan/data_pits_graph/template/lh.OASIS_testGrp_average.gii")
    args = parser.parse_args()

    path_to_folder = args.path_to_folder
    method = args.method
    path_mesh = args.path_mesh

    create_modified_graphs(path_to_folder, path_mesh, method)

        
    

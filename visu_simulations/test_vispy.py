import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
import vispy
from vispy.geometry import create_sphere
import tools.graph_visu as gv
import tools.graph_processing as gp
import slam.io as sio
import numpy as np
import networkx as nx
import vispy.scene.visuals as vs
from vispy.visuals import SurfacePlotVisual
import vispy.scene





file_path = 'example_mesh.gii'
test_mesh = sio.load_mesh(file_path)

vertices = test_mesh.vertices
faces = test_mesh.faces


# mesh = mesh.MeshVisual(vertices=vertices,faces=faces)

surf = SurfacePlotVisual(np.array(vertices))
canvas = vispy.scene.SceneCanvas(show=True)
view = canvas.central_widget.add_view()
view.add(surf)
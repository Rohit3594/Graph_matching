# Visualisation tools of multi-graph labelling of real world sulcal pits graphs

In this folder one can find the code necessary to visualise the labelling of the nodes obtained after using the multi-graph matching pipeline. There is different method of visualisation, either with points or with densities. There is also a cluster metric (silhouette metric) calculation tool available as well as the visualisation of this metric
In this folder one can find the code necessary to use multi-graph matching on real data. More specifically, one can find here a way to preprocess the graphs to make them usable by different algorithms, calculate the affinity and incidences matrices needed for KerGM, calculate the clustering to make the matching coherent and finally a way to postprocess the graph to make the visualisation of the clustering result possible.

## Installation procedure

In order to run the above code, one need to be sure to have set the right environment. We recommend to use Anaconda to do so. The depedency are as follow (latest version runs on python 3.6):

numpy:
```sh
pip install numpy
```

networkx:
```sh
pip install networkx
```

trimesh:
```sh
pip install trimesh
```

scipy:
```sh
pip install scipy
```

slam:
slam is a layer that comes over trimesh and which is used to sample random points on the surface of a sphere. To use it, one first needs to clone the project from https://github.com/gauzias/slam . 
In order to make the conda environement recognize slam as an external module, one need to add a file named slam_path.pth (or any name with the extension .pth) in path/to/conda/envs/{env-name}/lib/pythonX.X/site-packages/ .  The .pth file contains one line of text that gives the absolute path to the slam module : "path/to/slam" 

## Description of the different files




### script_visu_clustering.py

This script takes as an input a path to a folder where multigraph matching on graphs has been applied. It then shows on one median brain mesh all the labelled points where the color of the point indicates to which cluster it belongs and on a second mesh, it show all the points that are not labelled and are considered as noise by the pipeline. There is optional argument :

* --path_mesh: The path to the brain mesh to use.

### script_visu_density.py
This script calculate the density of each cluster (thanks to the heat equation applied on the laplacian of the graph) and show them on a mesh. Every cluster density is calculated separately and then the max value for each point of the mesh is shown. It also shows the density of the non labelled points. There are several optional arguments :
    
* --path_mesh: path to the brain mesh to use
* --path\_smoothed\_texture: path to a smoothed texture (representing the densities) already computed to avoid the smoothing calculation.
* --path\_to\_save: path where to save the smoothed texture once it is calculated. It will only happen if this argument is provided.

### script_visu_silhouette.py
This script calculate the silhouette value of each labelled point and propose different way of visualising it. First a method that match the visualisation in script\_visu\_clustering where each cluster centroid is a point where the color encode the silhouette value. Second, we replace the point in a flat texture by a sphere centered on the point with the color encoding the silhouette value. Be aware that with a lot of points calculating the silhouette value can take quite some time and we recommend to save the result once you have done it. Once again, several optional arguments exists:

* --path_mesh: The path to the brain mesh to use
* --path_silhouette: path to a file containing the silhouette information for each point
* --path\_to\_save: path where to save the silhouette information once calculated. Only used if the argument is provided.
* --method: the method to use to represent the silhouette in the centroid of each cluster. Either texture or spheres


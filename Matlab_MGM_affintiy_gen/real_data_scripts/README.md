# Scripts to run multi-graph matching pipeline on real world sulcal pits graphs

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

### script_preprocessing_graphs.py

This script takes as input a folder containing the graphs to use in the pipeline. It modifies the graph in a way that makes them compatible with the algorithms in the pipeline. These modified graphs (for example by adding geodesic distance information on the edges) are saved in a folder "modified\_graphs" such that the original graphs stays the same. It also create a python dictionary "correspondence\_dict" which allow to make the correspondence between the modified graphs (that are named with integer) and the original graph name. 

### script_affinity_calculation.py

This script generates the affinity (and incidence) matrices of all pair of preprocessed graphs in a given folder. These affinity matrices are the one needed by KerGM, which means that the full affinity matrices are not calculated (because the computational and storage overhead are too important). The affinty are obtained using a gaussian kernel with a gamma value calculated using a heuristic already used in the paper "Structural Graph-Based Morphometry: a multiscale searchlight framework based on sulcal pits" by Takerkart et al. The optional arguments are :
* --nb_workers: Decide on how many workers to launch in parallel. Each worker will generate the affinity matrices for a pair of graph

### script_generate_matching_with_clustering.py

This script takes as an input the folder were the preprocessed graphs have been matched using mALS. Starting from this matching, it will create a cost matrix based on how many GraphView match two given nodes together and run a clustering algorithm (DBSCAN) using this cost matrix. The result will be saved in the original folder. There are two optional arguments that correspond to the parameters of DBSCAN (more information about these parameter on the sklearn page of the algorithm):
* --param_eps: The eps parameter of DBSCAN. A float between 0 and 1 that represent the percent of the maximum distance observed in the cost matrix.
* --param\_min\_sample: the min\_sample parameter of DBSCAN. A float between 0 and 1 that represent the percent of the total number of graphs needed to consider a neighborhood a cluster. 

### script_postprocessing_graphs.py

This script postprocess the modified graphs to make them usable for future visualisation task. The main idea is to include the result of the clustering as node information in the graphs. It requires the folder where the graphs and the clustering results are.

### script\_create\_graphs\_from\_other\_methods

This script is usedto generate a set of sulcal pits graphs, that can later be used with visualisation tools, from a set of texture representing the labelling of the pits of each subject. It work with two state-of-the-art methods, the watershed clustering of Auzias et al in Neuroimage and the varifold labelling method by Kaltenmark et al in Media. The parameters are the following:

* --method: the name of the method used to generate the data, either neuroimage or media
* --path_mesh: path to the spherical mesh that correspond to the texture in the folder.

### full_process_graph_matching.sh

This bash script is here as a utility that makes the use of the pipeline slightly easier. One needs to modify in the file some variables such as the number of workers, the matlab path or the folder where this repo is and then by simply executing this file with the path to the folder with the graphs to match as the only argument (no relative path !) and the whole pipeline should just execute for these graphs.

### script_full_affinity_generation.py
This script is here to generate the full affinity matrices in order to calculate the value of different matching. However this feature is not done yet and is here as a first step to build it.



# Multi-graph matching algorithms

This module holds the necessary matlab files to run different multi-graph matching algorithms on our synthetic graphs and real-world graphs. Most of the code, especially the algorithms, are from the authors of these algorithms. The algorithms are CAO (with the different variants), mALS, mSync, mOpt.

## Description of useful files

### load_compute_and_save_multi_matching

This file will launch the different multi-graph matching algorithms based on one family of _n_ graphs. It saves the result in the same folder.

### Compare_multigraph_algorithms

Allow to go through different families of graphs to run the load_compute_and_save_multi_matching function.

### load_and_save_mALS_real_data.m

Go through a folder of real-data, get the bulk matrix from KerGM and generate the associated mALS results.



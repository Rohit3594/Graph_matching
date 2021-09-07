# Pairwise graph matching algorithms

This module contains the code to run several pairwise matching algorithms on either simulated graphs or real data. Most of the code (especially the different algorithms) in this module was not written by Nathan Buskulic but by the authors of the different algorithms. The algorithms are IPFP, SMAC, RRWM and KERGM.

## Description of useful files

### make\_1 and make\_2

Before anything else, one needs to run these two files to generate the appropriate function file needed by the algorithms.

### addPath

Run to add to the path the files necessary for running the different algorithms.

### Comparison_pairwise_algorithms

File used to generate the results of different pairwise graph matching algorithms on a set of synthetic sulcal pits graphs. More precisely, goes through every pair and call the launch_and_save_pairwise_matching_algorithms function.

### full_pairwise_for_multigraph

Function that generate a bulk matrix containing all the pairwise matching, done through exact KerGM, of a family of _n_ synthetic graphs generated for the multi-graph module.

### full_pairwise_real_data

Function that generate a bulk matrix containing all the pairwise matching, done through exact KerGM, for a set of real-world sulcal pits graphs.

### launch_and_save_pairwise_matching_algorithms
Function that can launch different algorithms and save their results for a folder of synthetic graphs.

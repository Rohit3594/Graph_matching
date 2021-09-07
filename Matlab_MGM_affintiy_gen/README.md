# Multi-graph matching on sulcal pits graphs for simulated and real-world data, 2020 internship of Nathan Buskulic at INT

You will find in this repository the code and the tools used to do multi-graph matching on sulcal pits. The project is separated in two main parts: simulated data and real-world data. The use of simulated data was necessary to evaluate the different algorithms that could have been used since there is no ground truth on the real-world data. Except for matlab implementation of pairwise and multi-graph matching algorithms, the entirety of the code has been produce by Nathan Buskulic during his Master 2 internship at INT, Marseille.

The codebase is separated in different "modules" to make it more accessible to newcomers. Each of these modules have a personalised readme that will give more specific information about the module and how it should be used. We will now introduced these modules

## generation_graphs

This module is all about the creation of simulated graphs and the analysis of different matching algorithms on them. The module is splitted in two different submodule, one for the analysis (and the generation of the associated graphs) of pairwise matching algorithms and one for the analysis (and also the generation of the graphs) of multi-graph matching algorithms.

## generation_pairwise

This is a module that allows to generate pairs of graphs to match to test different pairwise matching algorithm. The details of the algorithms are in the module _pairwise\_algorithms_. It also permit the generation of affinity matrices necessary for the matching algorithms.

## generation_multi

In this module are the script necessary to generate families of _n_ graphs and use multi-graph matching algorithms on them as well as clustering with DBSCAN.

## real_data_scripts

Set of scripts necessary to run the multi-graph matching pipeline on real data for both our pipeline and other methods.

## real_data_visu

Set of script to visualise the labelling of the pits obtained after from our pipeline or other methods.

## pairwise_algorithms

The matlab implementation of several pairwise matching algorithms.

## multigraph_algorithms

The matlab implementation of several multi-graph matching algorithms.

## visualisation

It is a module offering visualisation of the matching on simulated graphs using stereo projection of the points from a sphere. However this module has not been used or checked thoroughly and was more an experiment than a real thing.

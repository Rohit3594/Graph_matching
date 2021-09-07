%% Compute multigraph matching assignments for every family in folder structure
% add to path the right libraries
% need to give the right path as an input before using addpath - Really
% useful to do so ?
%clear all

%path = "/hpc/meca/users/buskulic.n/stage_nathan/generation_graphes/generation_multi/generated_graphs_medium/"
%path = "/Users/Nathan/stageINT/stage_nathan/generation_graphes/generation_multi/generated_graphs_small_wo_outliers_8/"
%method = "good guess"
%nb_graphs_to_use = []

%path = "/Users/Nathan/stageINT/stage_nathan/data_pits_graph/OtherExemples/"

load_and_save_mALS_real_data(path, "KerGM")
            

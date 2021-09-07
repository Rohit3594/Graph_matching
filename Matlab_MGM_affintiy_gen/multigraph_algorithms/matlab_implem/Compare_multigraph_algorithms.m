%% Compute multigraph matching assignments for every family in folder structure
% add to path the right libraries
% need to give the right path as an input before using addpath - Really
% useful to do so ?
%clear all

%path = "/hpc/meca/users/buskulic.n/stage_nathan/generation_graphes/generation_multi/generated_graphs_medium/"
%path = "/Users/Nathan/stageINT/stage_nathan/generation_graphes/generation_multi/generated_graphs_small_wo_outliers_8/"
path = '/hpc/meca/users/rohit/stage_nathan/generation_graphes/generation_multi/test_for_pairwise/'
%method = "good guess"
%nb_graphs_to_use = []

%% Go through every folder

D = dir(path);
D = D(~ismember({D.name}, {'.', '..'}));

for k = 1:length(D)
    if D(k).isdir == 1
        % for each set of parameters
        % Get the path of the param
        path_param = strcat(D(k).folder, "/", D(k).name);
        
       
        A = dir(path_param);
        A = A(~ismember({A.name}, {'.', '..'}));
        
        for k2 = 1:length(A)
           
            % for each run
            % Get the path of the run
            path_run = strcat(A(k2).folder, "/", A(k2).name)
            
            % Get the multigraph matching for this run
            if method == "KerGM" || method == "good guess"
                % If we just want to use all the graphs
                if ~exist('nb_graphs_to_use','var') || isempty(nb_graphs_to_use)
                    load_compute_and_save_multi_matching(path_run, method);
                else % If we want to use subsets of graphs
                    path_to_save_subgraphs = strcat(path_run,"/","results_sub_graphs");
                    if ~exist(path_to_save_subgraphs, 'dir')
                        mkdir(path_to_save_subgraphs);
                    end
                    for i_graphs = 1:length(nb_graphs_to_use)
                        elem_graphs = int32(nb_graphs_to_use(i_graphs));
                        load_compute_and_save_multi_matching(path_run, method, [], path_to_save_subgraphs, elem_graphs);
                    end
                end
                
            elseif method == "mix"
                path_to_save_mix = strcat(path_run,"/","results_mix")
                
                % If the result folder does not already exist
                if ~exist(path_to_save_mix, 'dir')
                    mkdir(path_to_save_mix)
                end
                
                % Launch the computation for different mix rates
                for i_mix = 1:10
                    load_compute_and_save_multi_matching(path_run, method, int32(i_mix*10), path_to_save_mix);
                end
            end
            
        end
    end
end

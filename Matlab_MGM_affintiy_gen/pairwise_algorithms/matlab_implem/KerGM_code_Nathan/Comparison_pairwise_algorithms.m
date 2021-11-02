%% let's try to go through some folder
% first let's define a starting point that we will parametrized later
path = '/hpc/meca/users/rohit/stage_nathan/generation_graphes/generation_pairwise/simu_graph/'
%path = "/Users/Nathan/stageINT/stage_nathan/generation_graphes/generation_pairwise/new_simus_complete/"
affinity_name = 'affinity.mat'
incidences_name = 'incidence_matrices.mat'

%%
%cd(path)
D = dir(path);
D = D(~ismember({D.name}, {'.', '..'}));

for k = 1:length(D)
    if D(k).isdir == 1
        k
        % for each set of parameters
        path_param = strcat(D(k).folder, '/', D(k).name);
        A = dir(path_param);
        A = A(~ismember({A.name}, {'.', '..'}));
        
        % for each pair of graph
        for k2 = 1:length(A)
            %path_graph = strcat(A(k).folder, "/", A(k).name)
            %B = dir(path_graph)
            %B = B(~ismember({B.name}, {'.', '..'}));
            
            % load the wanted file
            path_graph_pair = strcat(A(k2).folder, '/', A(k2).name,'/')
            
            done = launch_and_save_pairwise_matching_algorithms(path_graph_pair, affinity_name, incidences_name)
        end
        %cd("..")
        
    end
end

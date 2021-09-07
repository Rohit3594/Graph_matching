%% Compute KerGM paiwise assignment for every pair of graphs in every family
%path = "/Users/Nathan/stageINT/stage_nathan/generation_graphes/generation_multi/generated_graphs_test/"
path = '/hpc/meca/users/rohit/stage_nathan/generation_graphes/generation_multi/test_for_pairwise/'
affinity_name = 'affinity'
incidence_name = 'incidence'

%% Go through every folder

D = dir(path);
D = D(~ismember({D.name}, {'.', '..'}));

for k = 1:length(D)
    if D(k).isdir == 1
        % for each set of parameters
        % Get the path of the param
        path_param = strcat(path, '/', D(k).name);
        
        
        
        A = dir(path_param);
        A = A(~ismember({A.name}, {'.', '..'}));
        
        for k2 = 1:length(A)
           
            % for each run
            % Get the path of the run
            path_run = strcat(path_param, '/', A(k2).name);
            
            % define a variable used to reinitialise the full assignment matrix
            full_assign_init = 0    
            
            % Get the number of graph
            nb_graph = length(dir(strcat(path_run,'/graphs/*.gpickle')));
            
            % Create the full assignmenet matrix
            %nb_tot_nodes = nb_graph * nb_nodes
       
            
            % For each combination of graphs
            for graph_nb_1 = 0:nb_graph-1
                for graph_nb_2 = graph_nb_1 + 1:nb_graph-1
                    % Get the assignment matrix for this pair of graph
                    X = launch_and_save_KerGM_matching(path_run, int32(graph_nb_1), int32(graph_nb_2), affinity_name, incidence_name);
                    
                    if full_assign_init == 0
                        nb_nodes = size(X,1);
                        nb_tot_nodes = nb_graph * nb_nodes;
                        full_assignment_mat = zeros(nb_tot_nodes, nb_tot_nodes);
                        full_assign_init = 1;
                    end
              
                    % Load it in the full assignment matrix
                    a = int32(graph_nb_1*nb_nodes) + 1;
                    b = int32((graph_nb_1+1)*nb_nodes);
                    c = int32(graph_nb_2*nb_nodes) + 1;
                    d = int32((graph_nb_2+1)*nb_nodes);
                    full_assignment_mat(a:b,c:d) = X;
                end
            end
            % fill the permutation matrices we didnt calculated by using
            % transpose + identity matrix for the diagonal
            full_assignment_mat = full_assignment_mat + full_assignment_mat' + eye(size(full_assignment_mat));
            % save the matrix
            save(strcat(path_run,'/X_pairwise_kergm.mat'),'full_assignment_mat');
        end
        
    end
end

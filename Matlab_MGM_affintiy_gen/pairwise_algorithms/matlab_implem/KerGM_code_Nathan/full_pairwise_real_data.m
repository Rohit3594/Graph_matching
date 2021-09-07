%% Compute KerGM paiwise assignment for every pair of graphs in every family
%path = "/Users/Nathan/stageINT/stage_nathan/data_pits_graph/OtherExemples/"
affinity_name = "affinity"
incidence_name = "incidence"


% Get the number of graph
nb_graph = length(dir(strcat(path,"/modified_graphs/*.gpickle")));

full_assign_init = 0

% For each combination of graphs
for graph_nb_1 = 0:nb_graph-1
    for graph_nb_2 = graph_nb_1 + 1:nb_graph-1
        % Get the assignment matrix for this pair of graph
        X = launch_and_save_KerGM_matching(path, int32(graph_nb_1), int32(graph_nb_2), affinity_name, incidence_name);

        if full_assign_init == 0
            nb_nodes = size(X,1)
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
save(strcat(path,"/X_pairwise_kergm.mat"),"full_assignment_mat");

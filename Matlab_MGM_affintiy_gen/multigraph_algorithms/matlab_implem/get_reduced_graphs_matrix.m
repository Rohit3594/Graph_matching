function [reduced_mat] = get_reduced_graphs_matrix(rawMat,nb_graphs_total, nb_graphs)
%get_reduced_graphs_matrix Returns the pairwise matrix assignment with only
%the nb_graphs first graphs. Not useful if used with nb_graphs =
%nb_graphs_total

nb_nodes = size(rawMat,1) / nb_graphs_total;
reduced_mat = rawMat(1:nb_nodes * nb_graphs, 1:nb_nodes * nb_graphs);
end


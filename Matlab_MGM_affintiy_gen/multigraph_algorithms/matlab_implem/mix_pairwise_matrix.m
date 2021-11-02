function [mixed_pairwise_mat] = mix_pairwise_matrix(pairwise_KerGM,pairwise_good_guess, percent_good_guess_to_take, nb_graphs, nb_nodes)
%mix_pairwise_matrix 
%   Get the two pairwise matrices from KerGM and good guess calculations
%   and uses a given percent of good gues and the rest is KerGM matching.
%   percent_good_guess_to_take is a number between 0 and 1

% initialise the result matrix
mixed_pairwise_mat = pairwise_KerGM;

% get the number of swap to make
nb_swap = int32(percent_good_guess_to_take * (nb_graphs-1) * nb_graphs / 200)

% We create the matrix of pairs of graphs we calculated an affinity matrix
% for
possible_combination = zeros(2, (nb_graphs-1) * nb_graphs / 2);
index_combination = 1
for graph_1 = 1:nb_graphs-1
    for graph_2 = graph_1+1:nb_graphs
       possible_combination(:,index_combination) = [graph_1, graph_2];
       index_combination = index_combination + 1; 
    end 
end

% We randomize the order of the pairs
possible_combination = possible_combination(:,randperm(size(possible_combination, 2)));

% We get the select the number of columns we need to make the swaps
possible_combination = possible_combination(:,1:nb_swap);

% We swap the chosen content
for i_combination = 1:size(possible_combination,2)
    graph_1 = possible_combination(1,i_combination);
    graph_2 = possible_combination(2,i_combination);
    
    a = (graph_1 - 1) * nb_nodes + 1;
    b = (graph_1) * nb_nodes;
    c = (graph_2 - 1) * nb_nodes + 1;
    d = (graph_2) * nb_nodes;
    
    mixed_pairwise_mat(a:b,c:d) = pairwise_good_guess(a:b,c:d);
    mixed_pairwise_mat(c:d,a:b) = pairwise_good_guess(c:d,a:b);
end

end


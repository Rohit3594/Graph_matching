function affinity_cell = get_affinity_cell(path_to_folder, nb_graphs, nb_nodes)
    % Read all the affinity values in the
    % folder and stack them in a cell
    % structure as needed for the demo code of CAO
    
    affinity_cell = cell(nb_graphs, nb_graphs);

    
    % Go through all the computed affinity
    for graph_nb_1 = 0:nb_graphs-1
        for graph_nb_2 = graph_nb_1 + 1:nb_graphs-1
            load(strcat(path_to_folder,"/affinity_",int2str(int32(graph_nb_1)),"_",int2str(int32(graph_nb_2)),".mat"));
            
            % Check that the variable name is K
            affinity_cell{graph_nb_1+1, graph_nb_2+1} = full_affinity;
            
            affinity_cell{graph_nb_2+1, graph_nb_1+1} = inverse_affinity(full_affinity, nb_nodes);
            
        end
    end

end
function inversed_matrix = inverse_affinity(aff_mat, nb_nodes)

    inversed_matrix = aff_mat;
    line_perm = zeros(size(aff_mat));
    
    % We need to permutate the lines and the columns to get the right
    % values
    
    for a = 1:nb_nodes
        for i = 1:nb_nodes
            current_line = (a-1) * nb_nodes + i;
            line_to_swap = (i-1) * nb_nodes + a;
            
            % Fill the line_perm matrix
            line_perm(current_line, line_to_swap) = 1;
           
        end
    end

    % Do the permutation
    inversed_matrix = line_perm * inversed_matrix * line_perm';
end
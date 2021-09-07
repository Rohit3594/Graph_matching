%% Test pour run les algos de multigraph matching
function done = load_and_save_mALS_real_data(path_to_run, pairwise_method, mix_rate, path_to_save, nb_graphs_to_use)
%clear all
clear affinity

% get the necessary info
nb_graphs_total = length(dir(strcat(path_to_run, "/modified_graphs/*.gpickle")));

if pairwise_method == "mix"
    if ~exist('mix_rate','var') || isempty(mix_rate)
        fprintf("please define the mix rate you want to use when using mix pairwise")
    end
end

if ~exist('path_to_save','var') || isempty(path_to_save)
    path_to_save = path_to_run;
end

if ~exist('nb_graphs_to_use', 'var') || isempty(nb_graphs_to_use)
    nb_graphs_to_use = nb_graphs_total;
    suffix_graph = "";
else
    suffix_graph = strcat("_graphs_",int2str(nb_graphs_to_use));
end

% Determine the number of graphs to use
if nb_graphs_to_use < nb_graphs_total
    nb_graphs = nb_graphs_to_use;
else
    nb_graphs = nb_graphs_total;
end

path_to_aff = strcat(path_to_run,"/affinity");

% We get the path for the method we want
% Get the pairwise matrix of assignments given the selected method
if pairwise_method == "KerGM"
    path_to_pairwise = strcat(path_to_run, "/X_pairwise_kergm.mat");
    load(path_to_pairwise);
    rawMat = full_assignment_mat;
    % Only useful if nb_graphs < nb_graphs_total
    rawMat = get_reduced_graphs_matrix(rawMat,nb_graphs_total, nb_graphs);
    
    clear full_assignment_mat;
    name_suffix_to_save = suffix_graph;
    
elseif pairwise_method == "good guess"
    path_to_pairwise = strcat(path_to_run, "/X_pairwise_goodguess.mat");
    load(path_to_pairwise);
    rawMat = full_assignment_mat;
    % Only useful if nb_graphs < nb_graphs_total
    rawMat = get_reduced_graphs_matrix(rawMat,nb_graphs_total, nb_graphs);
    
    clear full_assignment_mat;
    name_suffix_to_save = strcat("_good_guess",suffix_graph);
    
elseif pairwise_method == "mix"
    
    % get both assignment matrix
    path_to_KerGM = strcat(path_to_run, "/X_pairwise_kergm.mat");
    load(path_to_KerGM);
    kerGM_assign = full_assignment_mat;
    % Only useful if nb_graphs < nb_graphs_total
    kerGM_assign = get_reduced_graphs_matrix(kerGM_assign,nb_graphs_total, nb_graphs);
    
    path_to_goodguess = strcat(path_to_run, "/X_pairwise_goodguess.mat");
    load(path_to_goodguess);
    goodguess_assign = full_assignment_mat;
    % Only useful if nb_graphs < nb_graphs_total
    goodguess_assign = get_reduced_graphs_matrix(goodguess_assign,nb_graphs_total, nb_graphs);
    
    % get the mixed matrix
    nb_nodes = size(goodguess_assign,1) / nb_graphs;
    rawMat = mix_pairwise_matrix(kerGM_assign, goodguess_assign, mix_rate, nb_graphs, nb_nodes);
    clear full_assignment_mat;
    
    % Decide of the name of the file
    name_suffix_to_save = strcat("_mix_",int2str(mix_rate),suffix_graph)
    
else
    fprintf("wrong method name. Accepted names are KerGM, good guess or mix");
end

%% Load the different data we need

% Get the number of graphs
%nb_graphs = length(dir(strcat(path_to_run, "/graphs/*.gpickle")));
nb_nodes = size(rawMat,1) / nb_graphs;

% Get the affinity values
%affinity.K = get_affinity_cell(path_to_aff, nb_graphs, nb_nodes);


%% Try to run the algorithms

% ========================================================
%Synchronize Permute from Pachauri et al
% if ~isfile(strcat(path_to_save,"/X_mSync",name_suffix_to_save,".mat"))
%     tic;
%     X = SynchronizePermute(rawMat,nb_nodes,nb_graphs,'sync');
%     t = toc;
%     save(strcat(path_to_save,"/X_mSync",name_suffix_to_save,".mat"), "X");
%     save(strcat(path_to_save,"/time_mSync",name_suffix_to_save,".mat"), "t");
% end
% ========================================================

% ========================================================
% MatchALS from Zhou et al
% Possibilité de régler pas mal de paramètres, mais avant on veut voir les
% paramètres par défaut.
if ~isfile(strcat(path_to_save,"/X_mALS",name_suffix_to_save,".mat")) || ~isfile(strcat(path_to_save,"/A_mALS",name_suffix_to_save,".mat"))
    tic;
    dim_group = zeros(nb_graphs,1) + nb_nodes;
    [X, A, info] = mmatch_CVX_ALS(rawMat,dim_group);
    X = full(X);
    t = toc;
    save(strcat(path_to_save,"/X_mALS",name_suffix_to_save,".mat"), "X");
    save(strcat(path_to_save,"/A_mALS",name_suffix_to_save,".mat"), "A");
    save(strcat(path_to_save,"/time_mALS",name_suffix_to_save,".mat"), "t");
end
% ========================================================

% ========================================================
%MatchOpt (star shaped), Yan et al
% tic;
% singleGraphConstList = cal_single_graph_consistency_score(rawMat,nb_nodes,nb_graphs);
% [C,refConstGraph] = max(singleGraphConstList);
% % given the reference graph r, first compute the unary
% % graph-wise consistency, then rank them into cstGrhList
% cstGrhList = rankGrhByConsistencyToRefGrh(rawMat,refConstGraph,nb_nodes,nb_graphs);
% updGrhList = [cstGrhList,refConstGraph];%for consistency rank
% refGraph = updGrhList(end);
% algpar.bPathSelect = 1;
% algpar.iccvIterMax = 1;
% algpar.iterMax1 = 10;
% algpar.bDisc = 0;
% algpar.algMethod = 'RRWM'; % PAS COOL A VOIR SI ON PEUT LE FAIRE UTILISER KERGM
% 
% X = ConsistMultiMatch(updGrhList,nb_nodes,nb_graphs,algpar,rawMat);
% t = toc;
% save(strcat(path_to_run,"/X_mOpt","_",name_suffix_to_save,".mat"), "X");
% save(strcat(path_to_run,"/time_mOpt","_",name_suffix_to_save,".mat"), "t");
% %========================================================
% 
% %========================================================
% iterRange = 10;
% scrDenom = 30;
% % pour le moment on dit qu'on a que des inliers
% % Test avec 80% inliers
% target.config.inCnt = int32(nb_nodes / 100 * 80);
% target.config.testType = 'formal';
% target.config.constIterImmune = 2; % in early iterations, not involve consistency, suggest 1-3
% target.config.constStep = 1.1;
% target.config.initConstWeight = 0.2;
% target.config.constWeightMax = 1;
% 
% tic;
% %CAO - Base - No smoothing - With inlier elicitation
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'afnty',1);
% t = toc;
% % In the cao name, if there is _s it means that smoothing was applied and
% % _o means that outliers elicitation was applied
% save(strcat(path_to_run,"/X_cao_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_o.mat"), "t");
% 
% %CAO - Base - With smoothing - With inlier elicitation
% X = SynchronizePermute(X,nb_nodes,nb_graphs,'sync');
% t = toc;
% save(strcat(path_to_run,"/X_cao_s_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_s_o.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - Base - No smoothing - Without inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'afnty',0);
% t = toc;
% save(strcat(path_to_run,"/X_cao.mat"), "X");
% save(strcat(path_to_run,"/time_cao.mat"), "t");
% %========================================================
% 
% %========================================================
% iterRange = 10;
% scrDenom = 30;
% % pour le moment on dit qu'on a que des inliers
% % Test avec 80% inliers
% target.config.inCnt = int32(nb_nodes / 100 * 80);
% target.config.testType = 'formal';
% target.config.constIterImmune = 2; % in early iterations, not involve consistency, suggest 1-3
% target.config.constStep = 1.1;
% target.config.initConstWeight = 0.2;
% target.config.constWeightMax = 1;

% %CAO - consistency - No smoothing - With inlier elicitation
% if ~isfile(strcat(path_to_save,"/X_cao_cst_o",name_suffix_to_save,".mat"))
%     tic;
%     X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'cstcy',1);
%     t = toc;
%     save(strcat(path_to_save,"/X_cao_cst_o",name_suffix_to_save,".mat"), "X");
%     save(strcat(path_to_save,"/time_cao_cst_o",name_suffix_to_save,".mat"), "t");
% end
% 
% %CAO - cst - With smoothing - With inlier elicitation
% X = SynchronizePermute(X,nb_nodes,nb_graphs,'sync');
% t= toc;
% save(strcat(path_to_run,"/X_cao_cst_s_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_cst_s_o.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - cst - No smoothing - Without inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'cstcy',0);
% t = toc;
% save(strcat(path_to_run,"/X_cao_cst.mat"), "X");
% save(strcat(path_to_run,"/time_cao_cst.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - UC - No smoothing - With inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'unary',1);
% t = toc;
% save(strcat(path_to_run,"/X_cao_uc_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_uc_o.mat"), "t");
% 
% %CAO - UC - With smoothing - With inlier elicitation
% X = SynchronizePermute(X,nb_nodes,nb_graphs,'sync');
% t= toc;
% save(strcat(path_to_run,"/X_cao_uc_s_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_uc_s_o.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - UC - No smoothing - Without inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'unary',0);
% t = toc;
% save(strcat(path_to_run,"/X_cao_uc.mat"), "X");
% save(strcat(path_to_run,"/time_cao_uc.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - PC - No smoothing - With inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'pair',1);
% t = toc;
% save(strcat(path_to_run,"/X_cao_pc_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_pc_o.mat"), "t");
% 
% %CAO - PC - With smoothing - With inlier elicitation
% X = SynchronizePermute(X,nb_nodes,nb_graphs,'sync');
% t = toc;
% save(strcat(path_to_run,"/X_cao_pc_s_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_pc_s_o.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - PC - No smoothing - Without inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'pair',0);
% t = toc;
% save(strcat(path_to_run,"/X_cao_pc.mat"), "X");
% save(strcat(path_to_run,"/time_cao_pc.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - C - No smoothing - With inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'exact',1);
% t=toc;
% save(strcat(path_to_run,"/X_cao_c_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_c_o.mat"), "t");
% 
% %CAO - C - With smoothing - With inlier elicitation
% X = SynchronizePermute(X,nb_nodes,nb_graphs,'sync');
% t=toc;
% save(strcat(path_to_run,"/X_cao_c_s_o.mat"), "X");
% save(strcat(path_to_run,"/time_cao_c_s_o.mat"), "t");
% %========================================================
% 
% %========================================================
% %CAO - C - No smoothing - Without inlier elicitation
% tic;
% X = CAO(rawMat,nb_nodes,nb_graphs,iterRange,scrDenom,'exact',0);
% t = toc;
% save(strcat(path_to_run,"/X_cao_c.mat"), "X");
% save(strcat(path_to_run,"/time_cao_c.mat"), "t");
% %========================================================

done = 1;
end

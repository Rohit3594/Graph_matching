function X = launch_and_save_distributed_KerGM_matching(path_to_folder, graph_nb_1, graph_nb_2, affinity_name, incidence_name)
    
    affinity_path = strcat(path_to_folder,"/affinity/",affinity_name,"_",graph_nb_1,"_",graph_nb_2,".mat")
    incidence_path = strcat(path_to_folder,"/affinity/",incidence_name,"_",graph_nb_1,"_",graph_nb_2,".mat")
    path_to_save = strcat(path_to_folder,"/KerGM_results/kergm_",graph_nb_1,"_",graph_nb_2,".mat")
    load(affinity_path);
    load(incidence_path);
    
    %algorithm parameter
    n1=size(G1,1); n2=size(G2,1);
    [pars, algs] = gmPar(2);
    Ct = ones(n1,n2); % mapping constraint (default to a constant matrix of one)
    KP=kN12;
    size(KP)
    %K = full_affinity;
    KQ = kE12;
    asgT.alg='truth';
    asgT.X=0;
    asgT.obj = 0;
    asgT.acc = 1;
    
    % KerGM
    graph1.G = G1;
    graph1.H = H1;
    graph1.K = kE11;

    graph2.G = G2;
    graph2.H = H2;
    graph2.K = kE22;

    lambda=0.005; num=11;
    [X, ~] = KerGM_Exact(graph1,graph2,KP,KQ,lambda,num);
    save(path_to_save,"X")
end
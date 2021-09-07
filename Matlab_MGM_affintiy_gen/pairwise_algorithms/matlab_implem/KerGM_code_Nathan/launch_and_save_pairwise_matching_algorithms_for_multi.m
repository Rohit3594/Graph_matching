
affinity_name = 'affinity'
incidences_name = 'incidence_matrices'

load(strcat(path_to_folder, affinity_name));
load(strcat(path_to_folder, incidences_name));
    
%algorithm parameter
n1=size(G1,1); n2=size(G2,1);
[pars, algs] = gmPar(2);
Ct = ones(n1,n2); % mapping constraint (default to a constant matrix of one)
KP=kN12;
K = full_affinity;
KQ = kE12;
asgT.alg='truth';
asgT.X=0;
asgT.obj = 0;
asgT.acc = 1;


% We launch the different algorithms
% IPFP-U
asgIpfpU = gm(K, Ct, asgT, pars{5}{:});
X = asgIpfpU.X;
save(strcat(path_to_folder,'X_ipf.mat'),'X');

% SMAC
asgSmac = gm(K, Ct, asgT, pars{4}{:});
X = asgSmac.X;
save(strcat(path_to_folder,'X_smac.mat'),'X');

% PM
%asgPm = pm(K, KQ, gphs, asgT);
%X = asgPm.X;
%save("/hpc/meca/users/buskulic.n/Echange_Pratiques/X_pm.mat",'X');

% RRWM
asgRrwm = gm(K, Ct, asgT, pars{7}{:});
X = asgRrwm.X;
save(strcat(path_to_folder,'X_rrwm.mat'),'X');

% KerGM
graph1.G = G1;
graph1.H = H1;
graph1.K = kE11;
    
graph2.G = G2;
graph2.H = H2;
graph2.K = kE22;

lambda=0.005; num=11;
[X, ~] = KerGM_Exact(graph1,graph2,KP,KQ,lambda,num);
save(strcat(path_to_folder,'X_kergm.mat'),'X');

done = 1
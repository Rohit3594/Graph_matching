#!/bin/bash

# Start time
start=$SECONDS

# We can put the conda environment here if it is necessary
# conda stuff !
#export PATH="/hpc/meca/users/buskulic.n/Applications/miniconda3/bin:$PATH"
#source activate slam_graph

# Definition of some variables
nb_workers=8
nb_workers_kergm=8
param_eps=0.5
param_min_sample=0.4

# Variable to the root folder of the project
#path_to_root_project=/hpc/meca/users/buskulic.n/stage_nathan/
path_to_root_project=/Users/Nathan/stageINT/stage_nathan/

# Get automatically the right script.
path_to_preprocessing="${path_to_root_project}real_data_scripts/script_preprocessing_graphs.py"
path_to_affinity="${path_to_root_project}real_data_scripts/script_affinity_calculation.py"
path_to_distributed_kergm="${path_to_root_project}real_data_scripts/script_distributed_kergm.py"
path_to_pairwise_kergm="${path_to_root_project}pairwise_algorithms/matlab_implem/KerGM_code_Nathan"
path_to_multigraph_mals="${path_to_root_project}multigraph_algorithms/matlab_implem"
path_to_clustering="${path_to_root_project}real_data_scripts/script_generate_matching_with_clustering.py"
path_to_postprocessing="${path_to_root_project}real_data_scripts/script_postprocessing_graphs.py"

#path_to_matlab=/hpc/soft/matlab/matlab2018b/bin/matlab
path_to_matlab=/Applications/MATLAB_R2018a.app/bin/matlab



# Launch the preprocessing of the graphs
#python $path_to_preprocessing "$1"

# Launch the affinity calculation
#python $path_to_affinity "$1" --nb_workers $nb_workers

# Launch the pairwise assignment calculation
#python $path_to_distributed_kergm "$1" --path_kergm $path_to_pairwise_kergm --path_matlab $path_to_matlab --nb_workers $nb_workers_kergm
#$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_pairwise_kergm}'));path='${1}';run('${path_to_pairwise_kergm}/full_pairwise_real_data.m');exit;"

# launch the multi-graph algorithm (mALS)
$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_multigraph_mals}'));path='${1}';method='KerGM';run('${path_to_multigraph_mals}/multigraph_real_data.m');exit;"

# Launch the clustering algorithm
python $path_to_clustering "$1" --param_eps $param_eps --param_min_sample $param_min_sample

# Postprocess the graphs
python $path_to_postprocessing "$1"

end=$SECONDS
duration=$(( end - start ))
echo "the script took $duration seconds"

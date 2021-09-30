#!/bin/bash
# Remove all the previously loaded modules

module purge

# Load all the available modules

module load all

# Load anacond3 for basic packages

module load anaconda/3

# Load python3 
module load python3

# Load matlab for the scripts
module load matlab

# Start time
start=$SECONDS

# We can put the conda environment here if it is necessary
# conda stuff !
#export PATH=/home/yadav.r/miniconda3/bin:$PATH
source activate slam_graph

# Definition of some variables
nb_runs=1
nb_graphs=3
nb_vertices=20
min_noise= 20
max_noise= 40
step_noise=20
min_outliers=0
max_outliers=5
step_outliers=5
cpt_full_matrix=0
nb_workers=24

# Variable to the root folder of the project
path_to_root_project=/hpc/meca/users/rohit/stage_nathan/
#path_to_root_project=/Users/Nathan/stageINT/stage_nathan/

# Get automatically the right script.
path_to_multi_graph_generation="${path_to_root_project}generation_graphes/generation_multi/script_generation_graphs_with_edges_permutation.py"
path_to_multi_affinity="${path_to_root_project}generation_graphes/generation_multi/script_generation_affinity_and_incidence_matrix.py"
path_to_pairwise_good_guess="${path_to_root_project}generation_graphes/generation_multi/script_get_pairwise_good_guess.py"

path_to_matlab=/hpc/soft/matlab/matlab2018b/bin/matlab
#path_to_matlab=/Applications/MATLAB_R2018a.app/bin/matlab

# Launch the generation of the graphs
#python $path_to_multi_graph_generation
echo "$1"
# Launch the calculation of the affinity matrix
python $path_to_multi_affinity "$1"  --nb_workers $nb_workers --cpt_full_matrix $cpt_full_matrix

# Launch the calculation of the pairwise asignment through matlab for KerGM
path_to_pairwise_algo="${path_to_root_project}pairwise_algorithms/matlab_implem/KerGM_code_Nathan"
$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_pairwise_algo}'));path='${1}';run('${path_to_pairwise_algo}/full_pairwise_for_multigraph.m');exit;"


#Launch the calculation of the pairwise assignment using good guess
python $path_to_pairwise_good_guess "$1" --nb_workers $nb_workers


# Launch the calculation of the multi-graph algorithms with KerGM as the pairwise basis
path_to_multigraph_algo="${path_to_root_project}multigraph_algorithms/matlab_implem"
$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_multigraph_algo}'));path='${1}';method='KerGM';run('${path_to_multigraph_algo}/Compare_multigraph_algorithms.m');exit;"


# Launch the calculation of the multi-graph algorithms with good guess as the pairwise basis
#$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_multigraph_algo}'));method='good guess';run('${path_to_multigraph_algo}/Compare_multigraph_algorithms.m');exit;"


# Launch the calculation of the multi-graph algorithms with a mixture of both good guess and KerGM
#$path_to_matlab -nodisplay -nosplash -nodesktop -r "addpath(genpath('${path_to_multigraph_algo}'));method='mix';run('${path_to_multigraph_algo}/Compare_multigraph_algorithms.m');exit;"

#Remove the affinities after computation
for dir in ${1}/*/
do
   #basename "$dir"
   echo ${dir}"0/affinity"
   rm -r ${dir}"0/affinity"
done


end=$SECONDS
duration=$(( end - start ))
echo "the script took $duration seconds"

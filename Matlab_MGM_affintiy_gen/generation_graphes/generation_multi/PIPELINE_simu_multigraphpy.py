# generate graphs
script_generation_graph.py
/hpc/meca/users/auzias/ISBI2020_graph_matching/simu/generation_multi/

# compute affinity and incidence matrix
conda activate trimesh_dev
python /hpc/meca/softs/dev/auzias/pyhon/stage_nathan/generation_graphes/generation_multi/script_generation_affinity_and_incidence_matrix.py /hpc/meca/users/auzias/ISBI2020_graph_matching/simu/generation_multi --nb_workers 32



#!/bin/bash
max=8

path_1=/home/rohit/PhD_Work/GM_my_version/Graph_matching/Matlab_MGM_affintiy_gen/useful_scripts
path_2=/home/rohit/PhD_Work/GM_my_version/Graph_matching/test_for_pairwise

for i in `seq 1 $max`
do
   sh ./test_2.sh "${path_2}_$i"
done
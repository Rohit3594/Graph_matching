import os
from os import listdir
from os.path import isfile, join

mypath = "./0/"

files_to_remove = ["graph_134.gpickle","graph_135.gpickle","graph_136.gpickle","graph_137.gpickle"]

directories = listdir(mypath)

for sub_folders in directories:

	onlyfiles = [f for f in listdir(mypath+sub_folders+"/graphs") if isfile(join(mypath+sub_folders+"/graphs", f))]

	for file in files_to_remove:

		os.remove(mypath+sub_folders+"/graphs/"+file)

	
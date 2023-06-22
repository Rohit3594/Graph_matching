import os
from torch_geometric.datasets import TUDataset
import os.path as osp
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool,dense_mincut_pool
import torch_geometric.transforms as T
from sklearn.model_selection import KFold

import sys
sys.path.append("/home/rohit/PhD_Work/GM_my_version/Graph_matching/")
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
from graph_generation.load_graphs_and_create_metadata import dataset_metadata
from graph_matching_tools.metrics import matching
import matplotlib.pyplot as plt
import scipy.io as sco
import slam.io as sio
from scipy.special import softmax
import pickle
from scipy.stats import betabinom
import seaborn as sns
import tools.graph_processing as gp
import tools.graph_visu as gv
from matplotlib.pyplot import figure
import pandas as pd
import random
from torch_geometric.utils.convert import from_networkx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.nn.functional import one_hot
from sklearn.preprocessing import OneHotEncoder
from torch.nn import Linear
import torch.nn.functional as F
from math import ceil
import torch_geometric as pyg
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import TopKPooling
from torch_geometric.data import Data
from torch_geometric.loader import DenseDataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.nn import GCNConv, DenseGraphConv


class topk(torch.nn.Module):
	
	def __init__(self, in_channels, out_channels, embed_dim=16):
		super(topk, self).__init__()
		
		
		self.conv1 = GraphConv(num_node_features, embed_dim)
		self.pool1 = TopKPooling(embed_dim, ratio=0.5,nonlinearity=torch.sigmoid)
		self.conv2 = GraphConv(embed_dim, embed_dim)
		self.pool2 = TopKPooling(embed_dim, ratio=0.5,nonlinearity=torch.sigmoid)

		self.lin1 = torch.nn.Linear(embed_dim * 2, embed_dim)
		self.bn1 = torch.nn.BatchNorm1d(embed_dim)

		self.lin2 = torch.nn.Linear(embed_dim, out_channels)


	def forward(self, x, edge_index, batch):
		

		x = F.relu(self.conv1(x, edge_index))
		x, edge_index_1, edge_attr_1, batch_1, perm_1, score_1 = self.pool1(x, edge_index, None, batch)
		x1 = torch.cat([gmp(x, batch_1), gap(x, batch_1)], dim=1)

		x = F.relu(self.conv2(x, edge_index_1))
		x, edge_index, _, batch, _, _ = self.pool2(x, edge_index_1, None, batch_1)
		x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)


		g_emb = x1 + x2
		#g_emb = torch.cat([x1,x2], dim=1)
		x = self.bn1(F.relu(self.lin1(g_emb)))
		x = F.dropout(x, p=0.5, training=self.training)
	
		x= F.dropout(x, p=0.5, training=self.training)
		out = F.log_softmax(self.lin2(x), dim=-1)
		
		return out, g_emb, self.pool1.weight, score_1
		

def graph_remove_dummy_nodes(graph):
	nodes_dummy_true = [x for x,y in graph.nodes(data=True) if y['is_dummy']==True]
	graph.remove_nodes_from(nodes_dummy_true)
	#print(len(graph.nodes))


def create_sulcal_dataset(path):

	list_graphs = gp.load_graphs_in_list(path)

	sulcal_dataset = []

	for i,g in enumerate(list_graphs):
		graph_remove_dummy_nodes(g) # remove dummy nodes
		g.remove_edges_from(nx.selfloop_edges(g)) # remove self loop edges
		
		attr_coords = np.array(list(nx.get_node_attributes(g,'sphere_3dcoords').values()))
		attr_basin_area = np.array(list(nx.get_node_attributes(g,'basin_area').values())).reshape([len(g),1])
		attr_basin_thickness = np.array(list(nx.get_node_attributes(g,'basin_thickness').values())).reshape([len(g),1])
		attr_depth = np.array(list(nx.get_node_attributes(g,'depth').values())).reshape([len(g),1])
		
		attr_concat = np.concatenate((attr_coords,attr_basin_area,attr_basin_thickness,attr_depth),axis = 1)
		
		#attr_concat = attr_coords
		
		x = torch.tensor(attr_concat,dtype=torch.float)
		
		#x = torch.tensor(nx.adjacency_matrix(g).todense(),dtype=torch.float)
		y = torch.tensor(graph_labels[i],dtype=torch.long)
		edge_index = torch.tensor(list(g.edges))
		
		sulcal_dataset.append(Data(x=x, y=y, edge_index=edge_index.t().contiguous()))

	return sulcal_dataset


def train():
	model.train()
	
	for data in train_loader:

		out, g_emb, w1,score_1 = model(data.x, data.edge_index, data.batch) 
		loss = criterion(out, data.y)  
		loss.backward() 
		optimizer.step() 
		optimizer.zero_grad()  

@torch.no_grad()
def test(loader):
	model.eval()
	correct = 0
	
	predictions = []
	
	for data in loader:
	
		out, g_emb, w1,score_1 =  model(data.x, data.edge_index, data.batch) 
		
		pred = out.argmax(dim=1)  
		predictions.append(pred)
		correct += int((pred == data.y).sum()) 
		
	test_acc = correct / len(loader.sampler)  
			
	return test_acc,model, predictions   


if __name__ == '__main__':


	path_to_labelled_graphs = '/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/Oasis_original_new_with_dummy/modified_graphs/'
	correspondence = pickle.load( open( "graph_correspondence_new.pickle", "rb" ) )
	Oasis_phen = pd.read_excel("/home/rohit/PhD_Work/GM_my_version/Graph_matching/data/OASIS_phenotype.ods", engine="odf")


	# Correspondence between sulcal graphs(OASIS) and gender. 

	#Oasis_phen[['Subject','M/F']]
	oasis_ids = Oasis_phen['Subject'].to_list()
	gender = Oasis_phen['M/F'].to_list()

	gender_corresp = []

	for corr in correspondence:
		corr_id = corr[0].split('_lh')
		
		for o_id, gen in zip(oasis_ids, gender):
			if o_id == corr_id[0]:
				gender_corresp.append([o_id,corr[1],gen])


	# Create graph level labels (here gender)
	onehot = OneHotEncoder(drop='first')
	graph_labels = onehot.fit_transform(np.array(gender_corresp)[:,2].reshape(-1,1)).toarray()


	sulcal_dataset = create_sulcal_dataset(path_to_labelled_graphs)
	num_node_features = sulcal_dataset[0].num_features
	print(len(sulcal_dataset))


	model = topk(num_node_features, 2)
	print(model)

	k = 8
	splits=KFold(n_splits=k,shuffle=True,random_state=42)


	history = {'train_acc':[],'test_acc':[]}

	for fold, (train_idx,test_idx) in enumerate(splits.split(np.arange(len(sulcal_dataset)))):
		
		print('Fold {}'.format(fold + 1))
		
		#model = GAT(hidden_channels=32)
		#model = MLP(num_node_features, 2)
		model = topk(num_node_features, 2)
		

		optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
		#criterion = torch.nn.CrossEntropyLoss()
		criterion = torch.nn.NLLLoss()
		
		
		train_sampler = SubsetRandomSampler(train_idx)
		test_sampler = SubsetRandomSampler(test_idx)
		
		train_loader = DataLoader(sulcal_dataset, batch_size=16, sampler=train_sampler)
		test_loader = DataLoader(sulcal_dataset, batch_size=16, sampler=test_sampler)
		
		
		train_acc_lst = []
		test_acc_lst = []

		best_accu = 0.0

		for epoch in range(1, 99):
			train()
			train_acc,_ ,_= test(train_loader)
			train_acc_lst.append(train_acc)

			test_acc, model,_ = test(test_loader)

			if train_acc > best_accu:
				
				if epoch > 20: 

					print('Saving Model ... ')
					#torch.save(model.state_dict(), 'OASIS_gender_cross_val_'+str(fold)+'.model')
					#torch.save(model.state_dict(), 'OASIS_MLP_gender_cross_val_'+str(fold)+'.model')
					torch.save(model.state_dict(), 'OASIS_topk_gender_cross_val_'+str(fold)+'.model')
					#best_accu = test_acc
					best_accu = train_acc

			test_acc_lst.append(test_acc)
			print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
			
			
		history['train_acc'].append(train_acc_lst)
		history['test_acc'].append(test_acc_lst)



	sns.set(rc={'figure.figsize':(11,6)})
	epochs = np.arange(1,epoch+1,1)

	kfold_mean_train = np.array(list(history['train_acc'])).mean(axis=0)
	kfold_std_train = np.array(list(history['train_acc'])).std(axis=0)

	kfold_mean_test = np.array(list(history['test_acc'])).mean(axis=0)
	kfold_std_test = np.array(list(history['test_acc'])).std(axis=0)



	plt.plot(epochs, kfold_mean_train ,label = 'avg training curve')
	plt.fill_between(epochs, kfold_mean_train - kfold_std_train, kfold_mean_train + kfold_std_train, alpha=0.2)


	plt.plot(epochs, kfold_mean_test ,label = 'avg test curve')
	plt.fill_between(epochs, kfold_mean_test - kfold_std_test, kfold_mean_test + kfold_std_test, alpha=0.2)


	plt.xlabel('Epoch',fontweight="bold",fontsize=15)
	plt.ylabel('Accuracy',fontweight="bold",fontsize=15)

	plt.xticks(fontsize=10)
	plt.yticks(np.arange(0.4,1,0.1),fontsize=10)

	plt.legend(loc = 'lower right')
	plt.show()






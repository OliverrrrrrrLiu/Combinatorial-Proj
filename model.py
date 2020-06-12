import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from preprocess import load_data

class GraphConvolutionLayer(nn.Module):
	"""Implements a simple GCN layer"""
	def __init__(self, fan_in, fan_out):
		"""
		@param fan_in: input dimension
		@param fan_out: output dimension
		"""
		super(GraphConvolutionLayer, self).__init__()
		self.fan_in = fan_in
		self.fan_out = fan_out
		self.weight = Parameter(torch.FloatTensor(fan_in, fan_out))
		self.init_parameters()

	def init_parameters(self):
		"""
		Initialize the parameter weight with Xavier init.
		"""
		std_dev = 1. / math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-std_dev, std_dev)

	def forward(self, input, adj_mtx):
		"""
		@param input: (hidden) representation to be passed into the GCN layer
		@param adj_mtx: adjacency matrix in torch sparse matrix format
		"""
		#output = torch.sparse.mm(adj_mtx, torch.mm(input, self.weight))+self.bias
		output = torch.sparse.mm(adj_mtx, torch.mm(input, self.weight))
		return output

class SymmetricBilinearLayer(nn.Module):
	"""Impelements a symmetric bilinear layer"""
	def __init__(self, dim):
		super(SymmetricBilinearLayer, self).__init__()
		self.dim = dim
		self.weight = Parameter(torch.FloatTensor(dim, dim))
		self.init_parameters()

	def init_parameters(self):
		std_dev = 1./math.sqrt(self.weight.size(1))
		self.weight.data.uniform_(-std_dev, std_dev)

	def forward(self, input):
		output = torch.mm(input, torch.mm((self.weight+self.weight.T)/2, input.T))
		return output

class GCNModel(nn.Module):
	"""Implements a GCN model for edge-wise optimal matching prediction"""
	def __init__(self, feature_dim, hidden_dims, out_dim, dropout = 0.5):
		"""
		@param feature_dim: dimension of feature space
		@param hidden_dims: list of hidden dimensions
		@param out_dim: output dimension of final GCN layer
		@param dropout: dropout probability, default: 0.5
		"""
		super(GCNModel, self).__init__()
		dims = [feature_dim] + hidden_dims + [out_dim]
		gcnlayers = [GraphConvolutionLayer(dims[i], dims[i+1]) for i in range(len(dims)-1)]
		self.gcnlayers = nn.Sequential(*gcnlayers)
		self.bilinear = SymmetricBilinearLayer(out_dim)
		self.dropout = dropout

	def forward(self, x, adj_mtx):
		"""
		Compute the forward pass. We assume a batch size of 1 for simplicity
		@param x: input node representation of dimension N*feature_dim
		@param adj_mtx: preprocessed adjacency matri of dimension N*N
		"""
		for i, layer in enumerate(self.gcnlayers):
			x = F.elu(layer(x, adj_mtx))
			if i < len(self.gcnlayers) - 1:
				x = F.dropout(x, self.dropout, training=self.training)
		out = torch.sigmoid(self.bilinear(x))
		return out

def loss_fn(out, idx):
	"""
	Compute the edge-wise classification loss
	@param out: predicted probabilities of edges
	@param idx: upper-triangular optimal matching edge index
	"""
	row, col = idx[:,0], idx[:,1]
	assert np.all(row < col)
	mask = torch.zeros(out.size()).float()
	mask[row, col] = 1.
	loss = -(100*torch.sum(mask*torch.log(out))+torch.sum(torch.triu((1-mask)*torch.log(1-out), diagonal=1)))
	return loss


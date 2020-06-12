import os
import torch
import numpy as np
import scipy.sparse as sp
from scipy import io


def load_data(path, feature_mode = "plain", return_orig = False):
    """Load data for preprocessing and training
    @param path: path to training/validation files
    @param feature_mode: mode for input feature
    @param return_orig: whether to return the original adjacency matrix, along with the preprocessed one
    """
    gpath = os.path.join(path, "graph")
    gnames = os.listdir(gpath)
    gnames.sort()
    mpath = os.path.join(path, "match")
    mnames = os.listdir(mpath)
    mnames.sort()
    data = []
    for i in range(len(gnames)):
        graph = np.genfromtxt(os.path.join(gpath, gnames[i])).astype(int)
        nodes = np.max(graph)+1
        label = np.genfromtxt(os.path.join(mpath, mnames[i])).astype(int)
        nodes = np.max(graph)+1 # number of vertices in graph
        idx = np.where(graph[:,0] != graph[:,1]) # identify self-loops
        row, col = graph[idx,0].flatten(), graph[idx,1].flatten()
        adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(nodes, nodes), dtype=np.float32)
        adj = adj + adj.T # construct symmetric adjacency matrix
        adj[adj > 0] = 1.
        if feature_mode == "plain":
            feat = torch.ones(nodes, 1).float()
        else: # feature_mode == "degree"
            feat = torch.Tensor(adj.sum(axis=1)/adj.shape[1]).view(-1, 1).float()
            feat = (feat-torch.min(feat))/(torch.max(feat)-torch.min(feat))
            #feat = (feat-torch.mean(feat))/torch.sqrt(torch.std(feat))
        adj_hat = normalize(adj + sp.eye(adj.shape[0]))
        adj_hat = sparse_mx_to_torch_sparse_tensor(adj_hat)
        if return_orig:
            data.append((feat, adj_hat, label, adj))
        else:
            data.append((feat, adj_hat, label))
    return data

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


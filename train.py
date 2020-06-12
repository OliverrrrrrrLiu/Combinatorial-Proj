import torch
import torch.optim as optim
import argparse
import numpy as np
import scipy.sparse as sp
from preprocess import load_data
from model import GCNModel, loss_fn


parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=False, help="Whether to train with GPU")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-3, help="Initial learning rate")
parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2-regularization parameter")
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
parser.add_argument("--feature_mode", type=str, default="plain", help="Feature mode")

args = parser.parse_args()
IS_CUDA = args.cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if IS_CUDA: torch.cuda.manual_seed(args.seed)

train_data = load_data("data/train", feature_mode=args.feature_mode)
eval_data = load_data("data/train", feature_mode=args.feature_mode, return_orig=True)

model = GCNModel(1, [16], 16, args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

if IS_CUDA: model.cuda()

def train_epoch(epoch_idx):
	"""Train one epoch"""
	loss_history = []
	model.train()
	# shuffle training data for sgd training
	np.random.shuffle(train_data)
	for i, (feat, adj_mtx, match_idx) in enumerate(train_data):
		if IS_CUDA:
			feat = feat.cuda()
			adj_mtx = adj_mtx.cuda()
		optimizer.zero_grad()
		out = model(feat, adj_mtx)
		loss = loss_fn(out, match_idx)
		loss_history.append((loss/len(match_idx)).item())
		loss.backward()
		optimizer.step()
	print("Epoch {}\t avg. loss: {:.4f}".format(epoch_idx+1, np.mean(loss_history)))


def predict():
	model.eval()
	outs = []
	for i, (feat, adj_hat, match_idx, adj) in enumerate(eval_data):
		if IS_CUDA:
			feat = feat.cuda()
			adj_mtx = adj_mtx.cuda()
		out = model(feat, adj_hat).data.numpy() # NxN
		mask = np.zeros(out.shape)
		mask[adj.nonzero()] = 1.
		out *= mask
		outs.append((out, len(match_idx)))
	return outs

def sort_idx_unravel(probs):
	count = np.count_nonzero(probs)
	i = probs.argsort(axis = None)[::-1]
	j = np.unravel_index(i, probs.shape)
	idx = np.vstack(j).T[:count]
	return idx[1::2] 

def greedy_matching(idx):
	node = []
	matching_count = 0
	for i in idx:
		u, v = i[0], i[1]
		if u in node or v in node:
			pass
		else:
			node = np.append(node,[u,v])
			matching_count += 1
	return matching_count

def evaluate():
	outs = predict()
	for graph_idx in range(len(outs)):
		probs, truth = outs[graph_idx]
		match = greedy_match(sort_idx_unravel(probs))
		print("Max-matching cardinality: {}\t Optimality gap: {:.4f}".format(match, (truth-match)/truth))


for _ in range(args.epochs):
	train_epoch(_)
	evaluate()

torch.save(model.state_dict(), "model.pt")

	




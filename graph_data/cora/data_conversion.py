# -*- coding:utf-8 -*-


import numpy as np
from utils import TrainHelp

name_index = {}
label_index = {}
idx_features_labels = np.genfromtxt("./cora.content", dtype=np.dtype(str))
for i in range(len(idx_features_labels[:, 0])):
    name_index[str(idx_features_labels[:, 0][i])] = i
all_labels = list(set(idx_features_labels[:, -1]))
for i in range(len(all_labels)):
    label_index[all_labels[i]] = i
feature = np.array(idx_features_labels[:, 1:-1], dtype=np.int)
labels = np.array([label_index[va] for va in idx_features_labels[:, -1]], dtype=np.int)
adj = np.zeros([feature.shape[0], feature.shape[0]], dtype=np.int)
cites = np.genfromtxt("./cora.cites", dtype=np.dtype(str))
for i in range(len(cites)):
    a, b = name_index[cites[i][0]], name_index[cites[i][1]]
    adj[a][b] = 1
    adj[b][a] = 1

adj, feature, labels = TrainHelp.get_max_connected_subgraph(adj, feature, labels)

np.savez("./cora_m.npz", adj=adj, feature=feature, labels=labels)

# -*- coding:utf-8 -*-



import numpy as np
from utils import TrainHelp

name_index = {}
label_index = {}
idx_features_labels = np.genfromtxt("./citeseer.content", dtype=np.dtype(str))
for i in range(len(idx_features_labels[:, 0])):
    name_index[str(idx_features_labels[:, 0][i])] = i
all_labels = list(set(idx_features_labels[:, -1]))
for i in range(len(all_labels)):
    label_index[all_labels[i]] = i
feature = np.array(idx_features_labels[:, 1:-1], dtype=np.int)
labels = np.array([label_index[va] for va in idx_features_labels[:, -1]], dtype=np.int)
adj = np.zeros([feature.shape[0], feature.shape[0]], dtype=np.int)
cites = np.genfromtxt("./citeseer.cites", dtype=np.dtype(str))
unfind = []
count = 0

for i in range(len(cites)):
    if cites[i][0] in name_index.keys() and cites[i][1] in name_index.keys():
        a, b = name_index[cites[i][0]], name_index[cites[i][1]]
        if a == b:
            pass
        else:
            adj[a][b] = 1
            adj[b][a] = 1
    else:
        if cites[i][0] not in name_index.keys():
            unfind.append(cites[i][0])
        if cites[i][1] not in name_index.keys():
            unfind.append(cites[i][1])
        count += 1

need_save = []
for i in range(adj.shape[0]):
    flag = False
    for j in range(adj.shape[1]):
        if adj[i][j] == 1:
            flag = True
            break
    if flag:
        need_save.append(i)
adj = adj[need_save, :][:, need_save]
feature = feature[need_save, :]
labels = labels[need_save]

adj, feature, labels = TrainHelp.get_max_connected_subgraph(adj, feature, labels)
np.savez("./citeseer_m.npz", adj=adj, feature=feature, labels=labels)






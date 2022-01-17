# -*- coding:utf-8 -*-
# @Time: 2021/9/9 14:24
# @Author: duiya duiyady@163.com


import numpy as np
from utils import TrainHelp


name_index = {}

idx_features_labels = []
for line in open("./Pubmed-Diabetes.NODE.paper.tab", "r"):
    idx_features_labels.append(line)
idx_features_labels.pop(0)
tmp_features = idx_features_labels[0].split("\t")
feature_index = {}
for i in range(len(tmp_features)-2):
    val = tmp_features[i+1].split(":")[1]
    feature_index[val.strip()] = i
idx_features_labels.pop(0)


feature = np.zeros([len(idx_features_labels), len(feature_index)], dtype=np.float32)
name_index = {}
labels = np.zeros(len(idx_features_labels), dtype=np.int)

for i in range(len(idx_features_labels)):
    tmp = idx_features_labels[i].split("\t")
    name_index[tmp[0].strip()] = i
    labels[i] = int(tmp[1].split("=")[1])-1
    for val in tmp[2: -1]:
        tmp_val = val.rsplit("=", 1)
        value = float(tmp_val[1])
        feature_name = tmp_val[0].strip()
        feature[i][feature_index[feature_name]] = value

adj = np.zeros([len(name_index), len(name_index)])

count = 0
for line in open("./Pubmed-Diabetes.DIRECTED.cites.tab", "r"):
    if count > 1:
        tmp = line.split("\t", 1)[1].replace("\t", "").replace("\n", "").split("|")
        i, j = name_index[tmp[0].split(":")[1].strip()], name_index[tmp[1].split(":")[1].strip()]
        adj[i][j] = 1
        adj[j][i] = 1
    count += 1

adj, feature, labels = TrainHelp.get_max_connected_subgraph(adj, feature, labels)

np.savez("./pubmed_m.npz", adj=adj, feature=feature, labels=labels)
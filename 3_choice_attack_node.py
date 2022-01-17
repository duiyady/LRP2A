# -*- coding:utf-8 -*-

import numpy as np
import pickle
import logging
import random

import os
import tensorflow as tf


seed = 2021
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


dataset = "cora"
model_name = "gat"

save_path = "./models/victim_model/" + dataset + "_" + model_name
black_model_data_path = "/label_train_val.npz"


logging.warning("数据集: " + dataset)
logging.warning("模型类别: " + model_name)
logging.warning("模型地址: " + save_path)
logging.warning("模型原始数据: " + black_model_data_path)




if dataset == "cora":
    data = np.load("./graph_data/cora/cora_m.npz")
elif dataset == "citeseer":
    data = np.load("./graph_data/citeseer/citeseer_m.npz")
elif dataset == "pubmed":
    data = np.load("./graph_data/pubmed/pubmed_m.npz")

ori_adj, ori_feature, ori_labels = data["adj"], data["feature"], data["labels"]
true_index = ori_labels



label_train_val = np.load(save_path + "/label_train_val.npz")
train_node = label_train_val["train_node"]
val_node = label_train_val["val_node"]
test_node = label_train_val["test_node"]

pre_index = label_train_val["pre_index"]
pre_softmax = label_train_val["pre_softmax"]



true_node = []
rate = []
label_node = {}
label_rate = {}
for node in test_node:
    if pre_index[node] == true_index[node]:
        if true_index[node] not in label_node.keys():
            label_node[true_index[node]] = []
            label_rate[true_index[node]] = []

        label_node[true_index[node]].append(node)
        label_rate[true_index[node]].append(pre_softmax[node][true_index[node]])
need_cha_node = []
high_node = []
low_node = []


for cla in label_node.keys():
    print(cla, len(label_node[cla]))
    if len(label_node[cla]) < 40:
        print(cla, "不要")
        continue
    rate, true_node = (list(t) for t in zip(*sorted(zip(label_rate[cla], label_node[cla]))))
    need_cha_node.extend(true_node[:40])
    need_cha_node.extend(true_node[-40:])
    high_node.extend(true_node[:40])
    low_node.extend(true_node[-40:])

all_s = {"high_node": high_node, "low_node": low_node}

pickle.dump(all_s, open(save_path + "/attack_node_置信度.pkl", "wb"))




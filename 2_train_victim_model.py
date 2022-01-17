# -*- coding:utf-8 -*-


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
from gnn_model.GAT import GAT
from gnn_model.GraphSage import GraphSage
from gnn_model.FastGCN import FastGCN
import utils.TrainHelp as TrainHelp
import random

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


seed = 2021
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


dataset = "cora"
model_name = "gat"
show = True

dims = [128]
samples = [20, 20]
if model_name == "graphsage":
    samples = [20, 20]
elif model_name == "fastgcn":
    samples = [128, 128]
Epoch = 1000

base_path = "./models/victim_model/" + dataset + "_" + model_name
victim_model_save_path = base_path + "/weight"
victim_model_pre_about = base_path + "/label_train_val.npz"



if dataset == "cora":
    data = np.load("./graph_data/cora/cora_m.npz")
elif dataset == "citeseer":
    data = np.load("./graph_data/citeseer/citeseer_m.npz")
elif dataset == "pubmed":
    data = np.load("./graph_data/pubmed/pubmed_m.npz")

ori_adj, ori_feature, ori_labels = data["adj"], data["feature"], data["labels"]
true_index = ori_labels
category_num = max(ori_labels) + 1
ori_labels = np.eye(max(ori_labels) + 1)[ori_labels]
train_node, val_node, test_node = TrainHelp.get_train_val_test_node(ori_labels, train_rate=0.1, val_rate=0.1)

ori_feature = ori_feature.astype(np.float32)


if model_name == "gat":
    victim_model = GAT(nfeat=ori_feature.shape[1], nclass=ori_labels.shape[1], alpha=0.2, nheads=2)
    victim_model.build(dims)

elif model_name == "graphsage":
    victim_model = GraphSage(nfeat=ori_feature.shape[1], nclass=ori_labels.shape[1])
    victim_model.build(dims, samples=samples)

elif model_name == "fastgcn":
    victim_model = FastGCN(nfeat=ori_feature.shape[1], nclass=ori_labels.shape[1])
    victim_model.build(dims)
    deal_adj = ori_adj.astype(np.float32)
    victim_model.init_sample(samples, ori_feature, deal_adj)



print("train a victim model")
max_acc = TrainHelp.train(victim_model, model_name=model_name, adj=ori_adj, feature=ori_feature, labels=ori_labels, train_node=train_node, val_node=val_node,
                Epoch=Epoch, save=victim_model_save_path, show=show, update=True)
print(max_acc)
victim_model.load_weights(victim_model_save_path)

victim_model_pre_softmax = TrainHelp.get_pre_softmax(victim_model, model_name=model_name, adj=ori_adj, feature=ori_feature)
victim_model_pre_index = np.argmax(victim_model_pre_softmax, axis=1)

print(np.mean(victim_model_pre_index == np.argmax(ori_labels, axis=1)))
train_acc = np.mean(victim_model_pre_index[train_node] == np.argmax(ori_labels, axis=1)[train_node])
val_acc = np.mean(victim_model_pre_index[val_node] == np.argmax(ori_labels, axis=1)[val_node])
test_acc = np.mean(victim_model_pre_index[test_node] == np.argmax(ori_labels, axis=1)[test_node])

print(train_acc, val_acc, test_acc)

np.savez(victim_model_pre_about, train_node=train_node, val_node=val_node, test_node=test_node, pre_index=victim_model_pre_index, pre_softmax=victim_model_pre_softmax)
with open(base_path + "/readme.txt", "w") as f:
    f.write("model_name: " + model_name + "\n")
    f.write("dataset: " + dataset + "\n")
    f.write("dims: ")
    f.writelines(str(dims) + "\n")
    f.write("samples: ")
    f.writelines(str(samples))

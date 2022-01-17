# -*- coding:utf-8 -*-



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from gnn_model.GAT import GAT
from gnn_model.GraphSage import GraphSage
from gnn_model.FastGCN import FastGCN
import pickle
import logging
import random
import utils.TrainHelp as graph_util

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
DIRECT = True
PER_TYPE = 1
PER_NUM = 0
attack_node_type = "置信度"
if dataset == "cora":
    data = np.load("./graph_data/cora/cora_m.npz")
elif dataset == "citeseer":
    data = np.load("./graph_data/citeseer/citeseer_m.npz")
elif dataset == "pubmed":
    data = np.load("./graph_data/pubmed/pubmed_m.npz")

ori_adj, ori_feature, ori_labels = data["adj"], data["feature"], data["labels"]
ori_feature = ori_feature.astype(np.float32)
ori_adj = ori_adj.astype(np.float32)
true_index = ori_labels

ori_labels = np.eye(max(ori_labels) + 1)[ori_labels]
degree = np.sum(ori_adj, axis=1)


base_path = "./models/victim_model/" + dataset + "_" + model_name
victim_model_data_path = "/label_train_val.npz"


label_train_val = np.load(base_path + "/label_train_val.npz")
train_node = label_train_val["train_node"]
val_node = label_train_val["val_node"]
victim_model_pre_index = label_train_val["pre_index"]
victim_model_pre_softmax = label_train_val["pre_softmax"]

if model_name in ["gat", "graphsage", "fastgcn"]:
    for line in open(base_path + "/readme.txt").readlines():
        line = line.replace("\n", "")
        if line.startswith("dims"):
            line = line[7: -1].split(",")
            dims = [int(val.replace(" ", "")) for val in line]
        elif line.startswith("samples"):
            line = line[10: -1].split(",")
            samples = [int(val.replace(" ", "")) for val in line]



if model_name == "gat":
    victim_model = GAT(nfeat=ori_feature.shape[1], nclass=ori_labels.shape[1], alpha=0.2, nheads=2)
    victim_model.build(dims)

elif model_name == "graphsage":
    victim_model = GraphSage(nfeat=ori_feature.shape[1], nclass=ori_labels.shape[1])
    victim_model.build(dims, samples=samples)
    adj_dict = graph_util.get_adj_dict(ori_adj)
    graphsage_A = victim_model.calculate_A(adj=ori_adj, max_degree=120)

elif model_name == "fastgcn":
    victim_model = FastGCN(nfeat=ori_feature.shape[1], nclass=ori_labels.shape[1])
    victim_model.build(dims)
    deal_adj = ori_adj.astype(np.float32)
    victim_model.init_sample(samples, ori_feature, deal_adj)



victim_model.load_weights(base_path + "/weight")



change_save_name = "/lrp2a_" + attack_node_type + "_"
if DIRECT:
    change_save_name = change_save_name + "direct_"
else:
    change_save_name = change_save_name + "undirect_"
if PER_TYPE == 1:
    change_save_name += "d+"
elif PER_TYPE == 2:
    change_save_name += "dx"
change_save_name = change_save_name + str(PER_NUM) + ".pkl"

logging.warning("数据集: " + dataset)
logging.warning("模型类别: " + model_name)
logging.warning("模型地址: " + base_path)
logging.warning("模型原始数据: " + victim_model_data_path)
logging.warning("对抗样本地址: " + change_save_name)


change_data = pickle.load(open(base_path + change_save_name, "rb"))


count = 0
success_count = 0
for ke in change_data:
    for node in tqdm(change_data[ke].keys()):
        count += 1
        new_graphsage_A = None
        if model_name == "graphsage":
            new_adj_dict = graph_util.get_adj_dict(ori_adj)
            new_graphsage_A = graphsage_A.copy()
            need_update = []
        change_struct = change_data[ke][node]

        if PER_TYPE == 0:
            if len(change_struct) > PER_NUM:
                change_struct = change_struct[: PER_NUM]


        change_feature = []
        tmp_adj = ori_adj.copy()
        tmp_feature = ori_feature.copy()


        for val in change_struct:
            if tmp_adj[val[0]][val[1]] == 0:
                tmp_adj[val[1]][val[0]] = 1
                tmp_adj[val[0]][val[1]] = 1
                if model_name == "graphsage":
                    need_update.append(val[0])
                    need_update.append(val[1])
                    new_adj_dict[val[0]].append(val[1])
                    new_adj_dict[val[1]].append(val[0])
            else:
                tmp_adj[val[1]][val[0]] = 0
                tmp_adj[val[0]][val[1]] = 0
                if model_name == "graphsage":
                    need_update.append(val[0])
                    need_update.append(val[1])
                    new_adj_dict[val[0]].remove(val[1])
                    new_adj_dict[val[1]].remove(val[0])

        if model_name == "graphsage":
            victim_model.change_A(new_graphsage_A, need_update, new_adj_dict, max_degree=120)
        elif model_name == "fastgcn":
            victim_model.change_adj(tmp_adj.astype(np.float32))


        change_pre = graph_util.get_pre_softmax(victim_model, model_name=model_name, adj=tmp_adj, feature=tmp_feature, node=node, graphsage_A=new_graphsage_A)
        change_pre_index = np.argmax(change_pre)


        if victim_model_pre_index[node] != change_pre_index:
            success_count += 1

print(success_count/count)



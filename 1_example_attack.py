# -*- coding:utf-8 -*-


import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from LRP2A_base import LRP2A
import numpy as np
from gnn_model.GAT import GAT
from gnn_model.GraphSage import GraphSage
from gnn_model.FastGCN import FastGCN
import utils.TrainHelp as TrainHelp
import matplotlib.pyplot as plt
import random
import os

seed = 2021
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


dataset = "cora"
model_name = "gat"
DIRECT = True
budget = 10
attack_node = 10
surrogate_dims = [200]
show = True

dims = [128]
samples = [20, 20]
if model_name == "graphsage":
    samples = [20, 20]
elif model_name == "fastgcn":
    samples = [128, 128]
Epoch = 10000


victim_model_save_path = "./models/victim_model/" + dataset + "_" + model_name + "/" + dataset + "_" + model_name
surrogate_model_save_path = "./models/surrogate_model/" + dataset + "_" + model_name + "/" + dataset + "_" + model_name
black_model_data_path = "/label_train_val.npz"



if dataset == "cora":
    data = np.load("./graph_data/cora/cora_m.npz")
elif dataset == "citeseer":
    data = np.load("./graph_data/citeseer/citeseer_m.npz")

ori_adj, ori_feature, ori_labels = data["adj"], data["feature"], data["labels"]
ori_adj = ori_adj.astype(np.float32)
ori_feature = ori_feature.astype(np.float32)
true_index = ori_labels
category_num = max(ori_labels) + 1
ori_labels = np.eye(max(ori_labels) + 1)[ori_labels]
train_node, val_node, test_node = TrainHelp.get_train_val_test_node(ori_labels, train_rate=0.1, val_rate=0.1)

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

print("victim model train")
TrainHelp.train(victim_model, model_name=model_name, adj=ori_adj, feature=ori_feature, labels=ori_labels, train_node=train_node, val_node=val_node,
                Epoch=Epoch, save=victim_model_save_path, show=False, update=True)
victim_model.load_weights(victim_model_save_path)

victim_model_pre_index = TrainHelp.get_pre(victim_model, model_name=model_name, adj=ori_adj, feature=ori_feature)
victim_model_pre_index = np.array(victim_model_pre_index)
victim_model_labels = np.eye(max(victim_model_pre_index)+1)[victim_model_pre_index]
ori_pre_soft = TrainHelp.get_pre_softmax(victim_model, model_name=model_name, adj=ori_adj, feature=ori_feature, node=attack_node)


print("surrogate model train")
lrp2a = LRP2A(ori_adj, ori_feature, class_num=ori_labels.shape[1], nhid=dims)
max_acc = lrp2a.train2(victim_model_labels, Epoch=Epoch, train_node=train_node, val_node=val_node, show=False, update=True,
                               save=surrogate_model_save_path)
print(max_acc)
lrp2a.surrogate_model.load_weights(surrogate_model_save_path)
lrp2a.attack_init()


print("start attack")
change_list = lrp2a.attack(t_node=attack_node, victim_label_index=victim_model_pre_index, per=budget,
                                              show=True, iter=500, ori_path=surrogate_model_save_path, direct=DIRECT)
new_adj = ori_adj.copy()
for val in change_list:
    if new_adj[val[0]][val[1]] == 0:
        new_adj[val[0]][val[1]] = 1
        new_adj[val[1]][val[0]] = 1
    else:
        new_adj[val[0]][val[1]] = 0
        new_adj[val[1]][val[0]] = 0

attack_pre_soft = TrainHelp.get_pre_softmax(victim_model, model_name=model_name, adj=new_adj, feature=ori_feature, node=attack_node)


index = np.arange(category_num)
a = 0.4
plt.bar(x=index, height=ori_pre_soft, width=a, color='darkorange', label=u'ori')
plt.bar(x=index+a+0.005, height=attack_pre_soft, width=a, color='crimson', label=u'attack')
font1 = {'family':'Times New Roman', 'weight':'normal', 'size':18,}
plt.legend(loc='upper left', prop={'size':10}, ncol=2)
plt.ylabel(u'probability', font1)
plt.xlabel(u'category', font1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
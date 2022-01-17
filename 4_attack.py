# -*- coding:utf-8 -*-



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import logging
import pickle
import random
import tensorflow as tf



seed = 2021
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

from LRP2A_base import LRP2A
from utils import TrainHelp
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

dataset = "cora"
model_name = "gat"
DIRECT = True
PER_TYPE = 1
PER_NUM = 0
attack_node_type = "置信度"
TRAIN_WEIGHT = True
base_path = "./models/victim_model/" + dataset + "_" + model_name
victim_model_pre_about = base_path + "/label_train_val.npz"
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

logging.warning("============ 基本信息 ================")
logging.warning("数据集: " + dataset)
logging.warning("攻击模型类别: " + model_name)
logging.warning("攻击模型地址: " + base_path)
logging.warning("攻击数据保存地址: " + change_save_name)
logging.warning("模型原始数据: " + victim_model_pre_about)

if dataset == "cora":
    data = np.load("./graph_data/cora/cora_m.npz")
elif dataset == "citeseer":
    data = np.load("./graph_data/citeseer/citeseer_m.npz")
elif dataset == "pubmed":
    data = np.load("./graph_data/pubmed/pubmed_m.npz")

ori_adj, ori_feature, ori_labels = data["adj"], data["feature"], data["labels"]
ori_feature = ori_feature.astype(np.float32)
degree = np.sum(ori_adj, axis=1)
degree = degree.astype(int)

true_index = ori_labels
ori_labels = np.eye(max(ori_labels) + 1)[ori_labels]

label_train_val = np.load(victim_model_pre_about)
val_node, victim_model_pre_index, train_node, test_node = label_train_val["val_node"], label_train_val["pre_index"], \
                                                          label_train_val["train_node"], label_train_val[
                                                              "test_node"]
victim_model_pre_index = np.array(victim_model_pre_index)
victim_model_labels = np.eye(max(victim_model_pre_index) + 1)[victim_model_pre_index]

labels = victim_model_labels

if attack_node_type == "度":
    attack_node = pickle.load(open(base_path + "/attack_node_度.pkl", "rb"))
elif attack_node_type == "置信度":
    attack_node = pickle.load(open(base_path + "/attack_node_置信度.pkl", "rb"))
elif attack_node_type == "随机":
    attack_node = pickle.load(open(base_path + "/attack_node_随机.pkl", "rb"))

print("train surrogate model.......")
dims = [200]
lrp2a_base_path = "../models/surrogate_model/LRP2A/" + dataset + "_" + model_name + "_" + "/weight"
lrp2a = LRP2A(ori_adj, ori_feature, class_num=ori_labels.shape[1], nhid=dims)
if TRAIN_WEIGHT:
    max_acc = lrp2a.train2(labels, Epoch=600, train_node=train_node, val_node=val_node, show=True, update=True, save=lrp2a_base_path)
    print(max_acc)
lrp2a.surrogate_model.load_weights(lrp2a_base_path)

lrp2a.attack_init()

adj_dict = TrainHelp.get_adj_dict(ori_adj)

change_data = {}
re = []
for ke in attack_node.keys():
    change_data[ke] = {}
    for now_index in range(len(attack_node[ke])):

        node = attack_node[ke][now_index]

        if degree[node] + PER_NUM <= 0:
            continue

        budget = PER_NUM
        if PER_TYPE == 1:
            budget += degree[node]
        elif PER_TYPE == 2:
            budget = degree[node] * PER_NUM



        info = ke + " 共: " + str(len(attack_node[ke])) + " 现在: " + str(now_index) + " node: " + str(
            int(attack_node[ke][now_index])) + " per: " + str(budget)
        logging.warning(info)

        change_list = lrp2a.attack(node, victim_label_index=victim_model_pre_index, per=budget,
                                          show=True, iter=500, ori_path=lrp2a_base_path, direct=DIRECT)

        change_data[ke][node] = change_list

        if now_index % 10 == 0 and now_index != 0:
            pickle.dump(change_data, open(base_path + change_save_name, "wb"))

    pickle.dump(change_data, open(base_path + change_save_name, "wb"))
pickle.dump(change_data, open(base_path + change_save_name, "wb"))

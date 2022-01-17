# -*- coding:utf-8 -*-


import threading
import numpy as np
import time
import os
from multiprocessing import Process, Queue, Value, Condition, Lock

class Producer_Batch(Process):
    def __init__(self, batch_node_queue, node, batch_size, is_end, cond, data_loader_flag, epoch_end_flag, shuffle=False):
        super(Producer_Batch, self).__init__()
        self.batch_node_queue = batch_node_queue
        self.node = node
        self.batch_size = batch_size
        self.cond = cond
        self.shuffle = shuffle
        self.is_end = is_end
        self.index = 0
        self.data_loader_flag = data_loader_flag
        self.epoch_end_flag = epoch_end_flag
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.node))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        self.is_end.value = 0
        self.index = 0

    def __len__(self):
        return int(np.floor((len(self.node)+self.batch_size-1) / self.batch_size))


    def run(self):
        while True:
            if self.epoch_end_flag.value == 1:
                break

            if self.data_loader_flag.value == 0:
                with self.cond:
                    self.cond.wait()

            indexes = self.indexes[self.index * self.batch_size:(self.index + 1) * self.batch_size]
            # 索引列表
            batch_node = [self.node[k] for k in indexes]
            self.batch_node_queue.put(batch_node)
            self.index = self.index+1
            if self.index == self.__len__():
                self.is_end.value = 1
                with self.cond:
                    self.cond.wait()
                self.on_epoch_end()



class Producer_Adj(Process):
    def __init__(self, batch_node_queue, batch_train_queue, adj_dict, feature, labels, nei_dis, batch_is_end, cond, sample_adj_flag, lock, data_loader_flag, epoch_end_flag, id=10, adj_type="gcn"):
        super(Producer_Adj, self).__init__()
        self.batch_node_queue = batch_node_queue
        self.batch_train_queue = batch_train_queue
        self.adj_dict = adj_dict
        self.feature = feature
        self.labels = labels
        self.nei_dis = nei_dis
        self.cond = cond
        self.t_id = id
        self.batch_is_end = batch_is_end
        self.sample_adj_flag = sample_adj_flag
        self.lock = lock
        self.epoch_end_flag = epoch_end_flag
        self.data_loader_flag = data_loader_flag
        self.adj_type = adj_type

    def get_all_node(self, train_node):
        all_node = []
        now_node = train_node
        for _ in range(self.nei_dis):
            tmp_node = []
            for c_node in now_node:
                if c_node not in all_node:
                    all_node.append(c_node)
                    tmp_node.extend(self.adj_dict[c_node])
            now_node = set(tmp_node)
        new_adj = np.zeros((len(all_node), len(all_node)), dtype=np.float32)

        if self.adj_type == "gcn":
            for i in range(len(all_node)):
                value = 1.0 / (len(self.adj_dict[all_node[i]]) + 1)
                new_adj[i][i] = value
                for j in range(len(all_node)):
                    if all_node[j] in self.adj_dict[all_node[i]]:
                        new_adj[i][j] = value

        elif self.adj_type == "self":
            for i in range(len(all_node)):  # 遍历某个连通图中的点

                nei_count = 0
                for nei in self.adj_dict[all_node[i]]:
                    if nei in all_node:
                        nei_count += 1

                # t_count = 1 + len(self.adj_dict[all_node[i]]) + 1
                t_count = 1 + nei_count + 1
                rate = 1 / t_count
                t_rate = 0.0
                if len(all_node) > 1 + nei_count:
                    t_rate = rate / (len(all_node) - 1 - nei_count)
                new_adj[i, :] = t_rate
                new_adj[i, i] = rate
                for j in range(len(all_node)):
                    if all_node[j] in self.adj_dict[all_node[i]]:
                        new_adj[i][j] = rate
        return new_adj, all_node

    def create_train_sample(self, node):
        # 生成数据
        new_adj, all_node = self.get_all_node(node)
        new_feature = self.feature[all_node, :].astype(np.float32)
        new_label = np.random.random([len(node), 7])
        if self.labels is not None:
            new_label = self.labels[node]
        self.batch_train_queue.put((new_adj, new_feature, new_label, node))

    def run(self):
        while True:
            if self.epoch_end_flag.value == 1:
                break
            if self.data_loader_flag.value == 0:
                with self.cond:
                    self.cond.wait()

            node = None
            try:
                node = self.batch_node_queue.get(timeout=1)
            except Exception as e:
                pass
            if node is not None:
                self.create_train_sample(node)
            else:
                if self.batch_is_end.value == 1:
                    with self.lock:
                        self.sample_adj_flag.value = self.sample_adj_flag.value + 1
                    with self.cond:
                        self.cond.wait()



class DataLoader(object):
    def __init__(self, adj_dict, feature, labels, train_node, nhid, batch_size=32, shuffle=False, thread_num=8, adj_type="gcn"):
        self.batch_node_queue = Queue(40)  # 一个batch的训练点
        self.batch_train_queue = Queue(40)  # 一个子图的数据
        self.batch_flag = Value("i", 1)  # 判断训练点是否生成完 初始为1
        self.sample_adj_flag = Value("i", 0)  # 有多少个生成子图进程结束了
        self.data_loader_flag = Value("i", 0)  # 保证初始化后一开始不运行
        self.epoch_end_flag = Value("i", 0)
        self.lock = Lock()  # adj_flag锁
        self.cond = Condition()
        self.batch_producer = Producer_Batch(self.batch_node_queue, train_node, batch_size, self.batch_flag, self.cond, self.data_loader_flag, self.epoch_end_flag, shuffle=shuffle)
        self.batch_producer.start()
        self.thread_num = thread_num
        self._workers = []
        self.iter_count = 0
        for i in range(thread_num):
            _worker = Producer_Adj(self.batch_node_queue, self.batch_train_queue, adj_dict, feature, labels, nhid, self.batch_flag, self.cond, self.sample_adj_flag, self.lock, self.data_loader_flag, self.epoch_end_flag, id=i, adj_type=adj_type)
            _worker.start()
            self._workers.append(_worker)


    def __iter__(self):
        self.iter_count += 1
        if self.iter_count == 1:
            self.data_loader_flag.value = 1
        if self.iter_count%2 == 0:
            with self.cond:
                self.cond.notify_all()
        return self

    def end_process(self):
        self.epoch_end_flag.value = 1
        with self.cond:
            self.cond.notify_all()

    def __next__(self):
        if self.sample_adj_flag.value != self.thread_num:
            tmp_data = None
            while True:
                try:
                    tmp_data = self.batch_train_queue.get(timeout=1)
                except Exception as e:
                    pass
                if tmp_data is None:
                    if self.sample_adj_flag.value != self.thread_num:
                    # if self.batch_flag.value == 0:
                        continue
                break
            if tmp_data is not None:
                return tmp_data
            else:
                self.sample_adj_flag.value = 0
                raise StopIteration
        else:
            tmp_data = None
            try:
                tmp_data = self.batch_train_queue.get(timeout=1)
            except Exception as e:
                pass
            if tmp_data is not None:
                return tmp_data
            else:
                self.sample_adj_flag.value = 0
                raise StopIteration


if __name__ == '__main__':
    base_path = "../models/victim_model/pubmed_gat"
    victim_model_pre_about = base_path + "/label_train_val.npz"
    data = np.load("../graph_data/pubmed/pubmed_m.npz")

    ori_adj, ori_feature, ori_labels = data["adj"], data["feature"], data["labels"]

    true_index = ori_labels
    ori_labels = np.eye(max(ori_labels) + 1)[ori_labels]
    degree = np.sum(ori_adj, axis=1)
    degree = degree.astype(int)

    label_train_val = np.load(victim_model_pre_about)
    val_node, victim_model_pre_index, train_node = label_train_val["val_node"], label_train_val["pre_index"], \
                                                   label_train_val["train_node"]
    victim_model_pre_index = np.array(victim_model_pre_index)
    labels = np.eye(max(victim_model_pre_index) + 1)[victim_model_pre_index]
    import utils.TrainHelp as TrainHelp
    adj_dict = TrainHelp.get_adj_dict(ori_adj)
    print(len(train_node), len(train_node)/32)
    data_load = DataLoader(adj_dict, ori_feature, labels, train_node, nhid=3, batch_size=32, thread_num=8, shuffle=True)
    count = 0
    s_t = time.time()
    for (new_adj, new_feature, new_label) in iter(data_load):
        print(count, new_adj.shape, new_feature.shape, new_label.shape)
        count = count+1
    e_t = time.time()

    print("over", e_t-s_t)

    count = 0
    s_t = time.time()
    for tmp in iter(data_load):
        print(count, tmp[0].shape, tmp[1].shape)
        count = count + 1
    e_t = time.time()
    print(e_t-s_t)
    data_load.end_process()



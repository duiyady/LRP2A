# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np
from utils import TrainHelp


class GraphSage(tf.keras.Model):
    # samples [4, 5, 6]表示三阶邻居采6个， 2阶邻居采5个，一阶采4个
    def __init__(self, nfeat, nclass, dtype=tf.float32, aggregator_type="mean", **kwargs):
        super(GraphSage, self).__init__(**kwargs)
        self.nfeat = nfeat
        self.nclass = nclass
        self.var_dtype = dtype
        self.aggregator_type = aggregator_type
        self.__dict__["convolution_layer"] = []

    def build(self, input_shape, samples, use_bias=False):
        self.samples = samples
        input_dim = self.nfeat
        if self.aggregator_type == "mean":
            for i in range(len(input_shape)):
                tmp = tf.keras.layers.Dense(units=input_shape[i], name="layer_" + str(i), use_bias=use_bias, activation="relu")
                tmp.build([2 * input_dim])
                setattr(self, "layer_" + str(i), tmp)
                input_dim = input_shape[i]
                self.convolution_layer.insert(0, getattr(self, "layer_" + str(i)))
            tmp = tf.keras.layers.Dense(units=self.nclass, name="layer_" + str(i), use_bias=use_bias, activation="softmax")
            tmp.build([2 * input_dim])
            setattr(self, "layer_" + str(i + 1), tmp)
            self.convolution_layer.insert(0, getattr(self, "layer_" + str(i + 1)))
            super(GraphSage, self).build([])

    def call(self, feature, train_node):
        layer_nodes = []  # 保存每一层每个节点参与聚合的邻居
        input_nodes = []  # 保存每一层输入
        idx_lookup = []  # 保存每一层输入点特征的新的id
        tmp_inputs = train_node

        # 每一层聚合的点
        for i in range(len(self.samples)):
            input_nodes.append(tmp_inputs)
            sample_nodes = self.sample_node_layer(tmp_inputs, self.samples[i])
            layer_nodes.append(sample_nodes)
            tmp_inputs = np.union1d(np.unique(sample_nodes), tmp_inputs)
            tmp_idx_lookup = np.zeros(self.neighbor_array.shape[0], dtype=np.int32)
            for i in range(len(tmp_inputs)):
                tmp_idx_lookup[tmp_inputs[i]] = i
            idx_lookup.append(tmp_idx_lookup)
        idx_lookup.pop()
        tmp_idx_lookup = np.zeros(self.neighbor_array.shape[0], dtype=np.int32)
        for i in range(len(tmp_idx_lookup)):
            tmp_idx_lookup[i] = i
        idx_lookup.append(tmp_idx_lookup)

        tmp_fea = feature
        for i in reversed(range(len(layer_nodes))):
            self_vec = tf.nn.embedding_lookup(tmp_fea, tf.nn.embedding_lookup(idx_lookup[i], input_nodes[i]))
            neigh_vec = tf.nn.embedding_lookup(tmp_fea, tf.nn.embedding_lookup(idx_lookup[i], layer_nodes[i]))
            neigh_vec = tf.reduce_mean(neigh_vec, axis=1)
            input_vec = tf.concat(values=[self_vec, neigh_vec], axis=1)
            tmp_fea = self.convolution_layer[i](input_vec)
        return tmp_fea

    def set_A(self, adj):
        self.neighbor_array = tf.constant(adj, dtype=tf.int32)

    def sample_node_layer(self, ids, num_samples):
        adj_lists = tf.nn.embedding_lookup(self.neighbor_array, ids)  # 寻找到目标点的邻居
        adj_lists = tf.transpose(tf.random.shuffle(tf.transpose(adj_lists)))  # 将邻居打乱
        adj_lists = tf.slice(adj_lists, [0, 0], [-1, num_samples])  # 选出一定数量的邻居
        return adj_lists

    def calculate_A(self, adj, max_degree):
        adj_dict = TrainHelp.get_adj_dict(adj)
        A_ = np.ones((len(adj_dict), max_degree), dtype=np.int) * len(adj_dict)
        for i in range(len(adj_dict)):
            # print("\r{:s}graphsage构造邻居矩阵 {:.2f}".format("*"*(i%10)+" "*(10-i%10), i/len(adj_dict)), end="")
            if i in adj_dict.keys():
                cols = adj_dict[i]
            else:
                cols = []
            if len(cols) == 0:
                continue
            if len(cols) > max_degree:
                cols = np.random.choice(cols, max_degree, replace=False)
            elif len(cols) < max_degree:
                cols = np.random.choice(cols, max_degree, replace=True)
            A_[i] = cols
        # print()
        return A_

    def change_A(self, A, need_change, adj_dict, max_degree):
        need_change = list(set(need_change))
        for node in need_change:
            if node in adj_dict.keys():
                cols = adj_dict[node]
            else:
                cols = []
            if len(cols) == 0:
                continue
            if len(cols) > max_degree:
                cols = np.random.choice(cols, max_degree, replace=False)
            elif len(cols) < max_degree:
                cols = np.random.choice(cols, max_degree, replace=True)
            A[node] = cols
        return A
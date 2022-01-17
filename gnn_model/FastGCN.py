# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np


class FastGCN(tf.keras.Model):
    def __init__(self, nfeat, nclass, dtype=tf.float32, **kwargs):
        super(FastGCN, self).__init__(**kwargs)
        self.nfeat = nfeat
        self.nclass = nclass
        self.var_dtype = dtype
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.__dict__["convolution_layer"] = []


    def calculate_A(self, adj):
        self.degree = np.sum(adj, axis=1)
        D_ = np.diag(1 / np.power(self.degree + 1, 0.5))
        A_ = np.dot(np.dot(D_, adj + np.identity(adj.shape[0])), D_)
        return A_.astype(np.float32)

    def build(self, input_shape):
        input_dim = self.nfeat
        for i in range(len(input_shape)):
            tmp = tf.Variable(self.initializer(shape=(input_dim, input_shape[i])), name="layer_" + str(i), dtype=self.var_dtype)
            setattr(self, "layer_" + str(i), tmp)
            input_dim = input_shape[i]
            self.convolution_layer.append(getattr(self, "layer_" + str(i)))
        self.out_weight = tf.Variable(self.initializer(shape=(input_dim, self.nclass)), name="layer_" + str(i), dtype=self.var_dtype)
        super(FastGCN, self).build([])

    def call(self, A, feature):
        tmp_feature = feature
        for i in range(len(self.convolution_layer)):
            tmp_feature = tf.matmul(tf.matmul(A[i], tmp_feature), self.convolution_layer[i])
            tmp_feature = tf.keras.activations.relu(tmp_feature)

        tmp_feature = tf.matmul(tf.matmul(A[-1], tmp_feature), self.out_weight)
        tmp_feature = tf.keras.activations.softmax(tmp_feature)
        return tmp_feature


    def init_sample(self, sample_size, feature, adj):
        self.sample_size = sample_size
        self.num_layers = len(sample_size)
        self.feature = feature
        self.adj = adj
        col_norm = np.linalg.norm(adj, axis=0)
        self.probs = col_norm/np.sum(col_norm)

    def change_adj(self, adj):
        self.adj = adj
        col_norm = np.linalg.norm(adj, axis=0)
        self.probs = col_norm / np.sum(col_norm)

    def sampling(self, nodes):
        all_support = [[]] * self.num_layers
        cur_out_nodes = nodes
        for layer_index in range(self.num_layers - 1, -1, -1):
            cur_sampled, cur_support = self._one_layer_sampling(cur_out_nodes, self.sample_size[layer_index])
            all_support[layer_index] = cur_support
            cur_out_nodes = cur_sampled

        sampled_X0 = self.feature[cur_out_nodes]
        return sampled_X0, all_support, 0

    def _one_layer_sampling(self, nodes, output_size):
        support = self.adj[nodes, :]
        neis = np.nonzero(np.sum(support, axis=0))[0]
        p1 = self.probs[neis]
        p1 = p1 / np.sum(p1)
        sampled = np.random.choice(np.array(np.arange(np.size(neis))), output_size, True, p1)

        u_sampled = neis[sampled]
        support = support[:, u_sampled]
        sampled_p1 = p1[sampled]
        support = np.dot(support, np.diag(1.0 / (sampled_p1 * output_size)))
        return u_sampled, support




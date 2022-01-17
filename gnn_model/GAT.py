# -*- coding:utf-8 -*-


import tensorflow as tf
import numpy as np



class GAT(tf.keras.Model):
    def __init__(self, nfeat, nclass, alpha=0.2, nheads=4, dtype=tf.float32, **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.nfeat = nfeat
        self.nclass = nclass
        self.alpha = alpha
        self.nheads = nheads
        self.var_dtype = dtype
        self.__dict__["attention_layers"] = []

    def build(self, input_shape):
        input_dim = self.nfeat
        self.nhid = input_shape
        for i in range(len(input_shape)):
            attentions = [GraphAttentionLayer(input_dim*self.nheads, input_shape[i], alpha=self.alpha, concat=True, dtype=self.var_dtype) for _ in range(self.nheads)]
            setattr(self, "attention_"+str(i), attentions)
            input_dim = input_shape[i]
            self.attention_layers.append(getattr(self, "attention_"+str(i)))
        self.output_layer = GraphAttentionLayer(input_dim*self.nheads, self.nclass, alpha=self.alpha, concat=False, dtype=self.var_dtype)
        super(GAT, self).build([])

    def calculate_A(self, adj):
        degree = np.sum(adj, axis=1)
        D_ = np.diag(1 / np.power(degree + 1, 0.5))
        A_ = np.dot(np.dot(D_, adj + np.identity(adj.shape[0])), D_)
        A_ = A_.astype(np.float32)
        return A_

    def calculate_adj(self, train_node, adj_dict):
        all_node = []
        now_node = train_node
        for _ in range(len(self.nhid)+2):
            tmp_node = []
            for c_node in now_node:
                if c_node not in all_node:
                    all_node.append(c_node)
                    if c_node in adj_dict.keys():
                        tmp_node.extend(adj_dict[c_node])
            now_node = tmp_node

        new_adj = np.zeros((len(all_node), len(all_node)), dtype=np.float32)
        for i in range(len(all_node)):
            for j in range(len(all_node)):
                if all_node[i] in adj_dict.keys():
                    if all_node[j] in adj_dict[all_node[i]]:
                        new_adj[i][j] = 1.0
        return new_adj, all_node

    def call(self, adj, feature):
        tmp_feature = tf.tile(feature, (1, self.nheads))
        for layers in self.attention_layers:
            tmp_feature = tf.concat([layer(None, tmp_feature, adj) for layer in layers], axis=1)
        out = tf.keras.activations.elu(self.output_layer(None, tmp_feature, adj))
        return tf.nn.softmax(out)



class GraphAttentionLayer(tf.keras.layers.Layer):

    def __init__(self, in_features, out_features, alpha, concat=True, dtype=tf.float32, **kwargs):
        super(GraphAttentionLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.initializer = tf.keras.initializers.GlorotUniform()
        self.leakyrelu = tf.keras.layers.LeakyReLU(self.alpha)
        self.var_type = dtype
        self.build([])


    def build(self, input_shape):
        self.W = tf.Variable(self.initializer(shape=(self.in_features, self.out_features), dtype=self.var_type), name="weights")
        self.a1 = tf.Variable(self.initializer(shape=(self.out_features, 1), dtype=self.var_type), name="a")
        self.a2 = tf.Variable(self.initializer(shape=(self.out_features, 1), dtype=self.var_type), name="a")
        super(GraphAttentionLayer, self).build([])

    def call(self, _, feature, adj):
        h = tf.matmul(feature, self.W)
        attn_for_self = tf.matmul(h, self.a1)
        attn_for_neighs = tf.matmul(h, self.a2)
        dense = attn_for_self + tf.transpose(attn_for_neighs)
        dense = self.leakyrelu(dense)
        mask = -10e9 * (1.0 - adj)
        dense += mask
        dense = tf.keras.activations.softmax(dense)
        dense = tf.keras.layers.Dropout(0.2)(dense)
        h = tf.keras.layers.Dropout(0.2)(h)
        node_features = tf.matmul(dense, h)

        if self.concat:
            return tf.keras.activations.elu(node_features)
        else:
            return node_features

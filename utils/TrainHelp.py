# -*- coding:utf-8 -*-



import numpy as np
import tensorflow as tf
import random
from utils.data_loader import DataLoader

def train(model, model_name, adj, feature, labels, train_node=None, val_node=None, learning_rate=1e-3, Epoch=100, save=None, show=False, update=True, batch_size=64, max_degree=120, graphsage_A = None):
    if train_node is None:
        train_node, val_node = get_train_val_node(labels)

    if model_name == "graphsage":
        if graphsage_A is None:
            A_ = model.calculate_A(adj, max_degree)
        else:
            A_ = graphsage_A
        model.set_A(A_)

    elif model_name == "gcn" or model_name == "sgc":
        A_ = model.calculate_A(adj)
        A_ = A_.astype(np.float32)

    feature = feature.astype(np.float32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    ca_loss_fn = tf.keras.losses.CategoricalCrossentropy()
    max_acc = 0.0
    acc_no_up = 0


    if model_name == "gcn" or model_name == "sgc":
        for epoch in range(Epoch):
            with tf.GradientTape() as tape:
                result = model.call(A_, feature)

                train_result = tf.gather(result, train_node)
                train_loss = ca_loss_fn(y_true=labels[train_node], y_pred=train_result)
            if update:
                grads = tape.gradient(train_loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
            val_result = tf.gather(result, val_node)
            val_loss = ca_loss_fn(y_true=labels[val_node], y_pred=val_result)
            y_pre = tf.argmax(result, axis=1)
            y_true = np.argmax(labels, axis=1)
            train_pre = tf.gather(y_pre, train_node)
            train_true = y_true[train_node]
            val_pre = tf.gather(y_pre, val_node)
            val_true = y_true[val_node]
            if show:
                print("iter{:3d}  train_acc: {:.6f} loss: {:.6f} val_acc: {:.6f} loss: {:.6f}".format(epoch, np.mean(train_pre == train_true), train_loss, np.mean(val_pre == val_true), val_loss))
            if np.mean(val_pre == val_true) > max_acc:
                max_acc = np.mean(val_pre == val_true)
                max_acc_train = np.mean(train_pre == train_true)
                acc_no_up = 0
                if save is not None:
                    model.save_weights(save)
            else:
                acc_no_up += 1
            if acc_no_up > 20:
                break
        return max_acc

    elif model_name == "gat":
        adj_dict = get_adj_dict(adj)
        train_node, val_node = np.array(train_node, dtype=np.int), np.array(val_node, dtype=np.int)
        train_util = TrainUtil(train_node=train_node, val_node=val_node, batch_size=batch_size, labels=labels)
        # train_data_loader = DataLoader(adj_dict, feature, labels, train_node, nhid=len(model.nhid)+2, batch_size=32, shuffle=True, thread_num=8)
        # val_data_loader = DataLoader(adj_dict, feature, labels, val_node, nhid=len(model.nhid)+2, batch_size=32, shuffle=True, thread_num=8)
        for epoch in range(Epoch):
            train_util.reset()
            batch_now = 0
            epoch_train_loss = tf.keras.metrics.Mean()
            epoch_train_acc = tf.keras.metrics.CategoricalAccuracy()
            while not train_util.end("train"):
            # for (train_adj, train_feature, train_label) in iter(train_data_loader):
                train_node, train_label = train_util.next_minibatch_feed_dict(type="train")
                train_adj, all_node = model.calculate_adj(train_node, adj_dict)
                train_feature = tf.nn.embedding_lookup(feature, all_node)
                need = [i for i in range(len(train_node))]
                # need = [i for i in range(train_label.shape[0])]
                with tf.GradientTape() as tape:
                    result = model.call(train_adj, train_feature)
                    result = tf.gather(result, need)
                    train_loss = ca_loss_fn(y_true=train_label, y_pred=result)
                if update:
                    grads = tape.gradient(train_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_train_loss(train_loss)
                epoch_train_acc(train_label, result)
                if show:
                    print("\r Epoch:{:3d} train_iter:{:4d} train_acc={:.6f} train_loss:{:.6f} ".format(epoch, batch_now,
                                                                                                       epoch_train_acc.result().numpy(),
                                                                                                       epoch_train_loss.result().numpy()),
                          end="")
                batch_now += 1
            if show:
                print()

            train_util.reset()
            batch_now = 0
            epoch_val_loss = tf.keras.metrics.Mean()
            epoch_val_acc = tf.keras.metrics.CategoricalAccuracy()

            while not train_util.end(type="val"):
            # for (val_adj, val_feature, val_label) in iter(val_data_loader):
                val_node, val_label = train_util.next_minibatch_feed_dict(type="val")
                val_adj, all_node = model.calculate_adj(val_node, adj_dict)
                val_feature = tf.nn.embedding_lookup(feature, all_node)
                need = [i for i in range(len(val_node))]
                # need = [i for i in range(val_label.shape[0])]
                result = model.call(val_adj, val_feature)
                result = tf.gather(result, need)
                loss = ca_loss_fn(y_true=val_label, y_pred=result)
                epoch_val_loss(loss)
                epoch_val_acc(val_label, result)
                if show:
                    print("\r             val_iter:{:4d}   val_acc={:.6f}   val_loss:{:.6f}".format(batch_now,
                                                                                                    epoch_val_acc.result().numpy(),
                                                                                                    epoch_val_loss.result().numpy()),
                          end="")
                batch_now += 1
            if show:
                print()
            if epoch_val_acc.result().numpy() > max_acc:
                max_acc = epoch_val_acc.result().numpy()
                acc_no_up = 0
                if save is not None:
                    model.save_weights(save)
            else:
                acc_no_up += 1
            if acc_no_up > 20:
                break
        # train_data_loader.end_process()
        # val_data_loader.end_process()
        return max_acc
    elif model_name == "graphsage":
        train_node, val_node = np.array(train_node, dtype=np.int), np.array(val_node, dtype=np.int)
        train_util = TrainUtil(train_node=train_node, val_node=val_node, batch_size=batch_size, labels=labels)

        for epoch in range(Epoch):
            train_util.reset()
            batch_now = 0
            epoch_train_loss = tf.keras.metrics.Mean()
            epoch_train_acc = tf.keras.metrics.CategoricalAccuracy()
            while not train_util.end("train"):
                train_node, train_label = train_util.next_minibatch_feed_dict(type="train")
                with tf.GradientTape() as tape:
                    result = model.call(feature, train_node)
                    train_loss = ca_loss_fn(y_true=train_label, y_pred=result)
                if update:
                    grads = tape.gradient(train_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_train_loss(train_loss)
                epoch_train_acc(train_label, result)
                if show:
                    print("\r Epoch:{:3d} train_iter:{:4d} train_acc={:.6f} train_loss:{:.6f} ".format(epoch, batch_now, epoch_train_acc.result().numpy(), epoch_train_loss.result().numpy()), end="")
                batch_now += 1
            if show:
                print()

            train_util.reset()
            batch_now = 0
            epoch_val_loss = tf.keras.metrics.Mean()
            epoch_val_acc = tf.keras.metrics.CategoricalAccuracy()

            while not train_util.end(type="val"):
                val_node, val_label = train_util.next_minibatch_feed_dict(type="val")
                result = model.call(feature, val_node)
                loss = ca_loss_fn(y_true=val_label, y_pred=result)
                epoch_val_loss(loss)
                epoch_val_acc(val_label, result)
                if show:
                    print("\r             val_iter:{:4d}   val_acc={:.6f}   val_loss:{:.6f}".format(batch_now, epoch_val_acc.result().numpy(), epoch_val_loss.result().numpy()), end="")
                batch_now += 1
            if show:
                print()
            if epoch_val_acc.result().numpy() > max_acc:
                max_acc = epoch_val_acc.result().numpy()
                acc_no_up = 0
                if save is not None:
                    model.save_weights(save)
            else:
                acc_no_up += 1
            if acc_no_up > 10:
                break
        return max_acc
    elif model_name == "fastgcn":
        train_node, val_node = np.array(train_node, dtype=np.int), np.array(val_node, dtype=np.int)
        train_util = TrainUtil(train_node=train_node, val_node=val_node, batch_size=batch_size, labels=labels)
        for epoch in range(Epoch):
            train_util.reset()
            batch_now = 0
            epoch_train_loss = tf.keras.metrics.Mean()
            epoch_train_acc = tf.keras.metrics.CategoricalAccuracy()
            while not train_util.end("train"):
                train_node, train_label = train_util.next_minibatch_feed_dict(type="train")
                sampled_feats, sampled_adjs, var_loss = model.sampling(train_node)
                with tf.GradientTape() as tape:
                    result = model.call(sampled_adjs, sampled_feats)
                    train_loss = ca_loss_fn(y_true=train_label, y_pred=result)
                if update:
                    grads = tape.gradient(train_loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                epoch_train_loss(train_loss)
                epoch_train_acc(train_label, result)
                if show:
                    print("\r Epoch:{:3d} train_iter:{:4d} train_acc={:.6f} train_loss:{:.6f} ".format(epoch, batch_now, epoch_train_acc.result().numpy(), epoch_train_loss.result().numpy()), end="")
                batch_now += 1
            if show:
                print()
            train_util.reset()
            batch_now = 0
            epoch_val_loss = tf.keras.metrics.Mean()
            epoch_val_acc = tf.keras.metrics.CategoricalAccuracy()

            while not train_util.end(type="val", new_batch=1):
                val_node, val_label = train_util.next_minibatch_feed_dict(type="val", new_batch=1)
                sampled_feats, sampled_adjs, var_loss = model.sampling(val_node)
                result = model.call(sampled_adjs, sampled_feats)
                loss = ca_loss_fn(y_true=val_label, y_pred=result)
                epoch_val_loss(loss)
                epoch_val_acc(val_label, result)
                if show:
                    print("\r             val_iter:{:4d}   val_acc={:.6f}   val_loss:{:.6f}".format(batch_now, epoch_val_acc.result().numpy(), epoch_val_loss.result().numpy()), end="")
                batch_now += 1
            if show:
                print()

            acc = epoch_val_acc.result().numpy()
            if acc > max_acc:
                max_acc = acc
                acc_no_up = 0
                if save is not None:
                    model.save_weights(save)
            else:
                acc_no_up += 1
            if acc_no_up >= 50:
                break

def get_pre(model, model_name, adj, feature, node=None, max_degree=120, batch=64, graphsage_A=None):
    if model_name == "graphsage":
        if graphsage_A is None:
            A_ = model.calculate_A(adj, max_degree)
        else:
            A_ = graphsage_A
        model.set_A(A_)
    feature = feature.astype(np.float32)

    if model_name == "gat":
        adj_dict = get_adj_dict(adj)
        if node is not None:
            train_adj, all_node = model.calculate_adj([node], adj_dict)
            train_feature = tf.nn.embedding_lookup(feature, all_node)
            result = model.call(train_adj, train_feature)
            result = tf.argmax(result, axis=1).numpy().tolist()
            return result[node]
        else:
            result = []
            now = 0
            while now * batch < adj.shape[0]:
                start, end = now * batch, min((now + 1) * batch, adj.shape[0])
                node = list(range(start, end))
                train_adj, all_node = model.calculate_adj(node, adj_dict)
                train_feature = tf.nn.embedding_lookup(feature, all_node)
                need = [i for i in range(len(node))]
                tmp = model.call(train_adj, train_feature)
                tmp = tf.gather(tmp, need)
                tmp = tf.argmax(tmp, axis=1).numpy().tolist()
                result.extend(tmp)
                now += 1
            return result

    elif model_name == "graphsage":
        if node is not None:
            result = tf.argmax(model.call(feature, [node]), axis=1).numpy().tolist()
            return result[0]
        else:
            result = []
            now = 0
            while now * batch < adj.shape[0]:
                start, end = now * batch, min((now + 1) * batch, adj.shape[0])
                node = list(range(start, end))
                tmp = tf.argmax(model.call(feature, node), axis=1).numpy().tolist()
                result.extend(tmp)
                now += 1
            return result
    elif model_name == "fastgcn":
        if node is not None:
            sampled_feats, sampled_adjs, var_loss = model.sampling([node])
            result = model.call(sampled_adjs, sampled_feats)
            result = tf.argmax(result, axis=1).numpy().tolist()
            return result[0]
        else:
            result = []
            for now_index in range(adj.shape[0]):
                sampled_feats, sampled_adjs, var_loss = model.sampling([now_index])
                tmp = model.call(sampled_adjs, sampled_feats)
                tmp = tf.argmax(tmp, axis=1).numpy().tolist()
                result.extend(tmp)
            return result
    elif model_name == "gcn":
        A_ = model.calculate_A(adj)
        A_ = A_.astype(np.float32)
        result = model.call(A_, feature)
        result = tf.argmax(result, axis=1).numpy().tolist()
        if node is not None:
            result = result[node]
        return result

def get_pre_softmax(model, model_name, adj, feature, node=None, max_degree=120, batch=64, graphsage_A=None):
    if model_name == "graphsage":
        if graphsage_A is None:
            A_ = model.calculate_A(adj, max_degree)
        else:
            A_ = graphsage_A
        model.set_A(A_)
    feature = feature.astype(np.float32)

    if model_name == "gat":
        adj_dict = get_adj_dict(adj)
        if node is not None:
            train_adj, all_node = model.calculate_adj([node], adj_dict)
            train_feature = tf.nn.embedding_lookup(feature, all_node)
            need = [0, ]
            result = model.call(train_adj, train_feature).numpy()
            # result = tf.gather(result, need).numpy()
            return result[0]
        else:
            result = None
            now = 0
            while now * batch < adj.shape[0]:
                start, end = now * batch, min((now + 1) * batch, adj.shape[0])
                node = list(range(start, end))
                train_adj, all_node = model.calculate_adj(node, adj_dict)
                train_feature = tf.nn.embedding_lookup(feature, all_node)
                need = [i for i in range(len(node))]
                tmp = model.call(train_adj, train_feature)
                tmp = tf.gather(tmp, need)
                if result is None:
                    result = tmp
                else:
                    result = np.concatenate((result, tmp), axis=0)
                now += 1
            return result
    elif model_name == "graphsage":
        if node is not None:
            result = model.call(feature, [node]).numpy()
            return result[0]
        else:
            result = None
            now = 0
            while now * batch < adj.shape[0]:
                start, end = now * batch, min((now + 1) * batch, adj.shape[0])
                node = list(range(start, end))
                tmp = model.call(feature, node).numpy()
                if result is None:
                    result = tmp
                else:
                    result = np.concatenate((result, tmp), axis=0)
                now += 1
            return result
    elif model_name == "fastgcn":
        if node is not None:
            sampled_feats, sampled_adjs, var_loss = model.sampling([node])
            result = model.call(sampled_adjs, sampled_feats).numpy()
            return result[0]
        else:
            result = None
            for now_index in range(adj.shape[0]):
                sampled_feats, sampled_adjs, var_loss = model.sampling([now_index])
                tmp = model.call(sampled_adjs, sampled_feats).numpy()

                if result is None:
                    result = tmp
                else:
                    result = np.concatenate((result, tmp), axis=0)
            return result



# 获取最大连通子图  adj, feature, labels 1
def get_max_connected_subgraph(adj, feature, labels, show=False):
    all_conn = get_all_connected_subgraph(adj, show=show)
    result = all_conn[0]
    for val in all_conn:
        if len(val) > len(result):
            result = val
    return adj[result, :][:, result], feature[result], labels[result]

# 获取所有连通子图的节点 [[子图包含的节点],[],[], ....] 1
def get_all_connected_subgraph(adj=None, adj_dict=None, show=False):
    if adj_dict is None:
        adj_dict = get_adj_dict(adj)
    all_node = list(np.arange(len(adj_dict)))
    result = []
    while len(all_node) > 0:
        if show:
            print("\rgraph_util 获取连通子图还剩{:d}".format(len(all_node)), end="")
        gram_node = [all_node[0]]
        tmp_result = [all_node[0]]
        all_node.remove(all_node[0])

        while len(gram_node) > 0:
            tmp_gram = set()
            for tmp_node in gram_node:
                for check in adj_dict[tmp_node]:
                    if check not in tmp_result:
                        tmp_gram.add(check)
                        tmp_result.append(check)
                        all_node.remove(check)
            gram_node = list(tmp_gram)

        result.append(tmp_result)
    if show:
        print()
    return result



def get_train_val_test_node(labels, train_rate=0.1, val_rate=0.1):
    train_node = []
    val_node = []
    test_node = []
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=1)
    label_index = {}
    for i in range(len(labels)):
        if labels[i] in label_index.keys():
            label_index[labels[i]].append(i)
        else:
            label_index[labels[i]] = [i]

    for class_type in label_index.keys():
        random.shuffle(label_index[class_type])
        tmp_train = []
        tmp_val = []
        tmp_test = []
        class_type_count = len(label_index[class_type])
        for i in range(class_type_count):
            if i <= int(class_type_count*train_rate):
                tmp_train.append(label_index[class_type][i])
            elif i <= int(class_type_count*(train_rate+val_rate)):
                tmp_val.append(label_index[class_type][i])
            else:
                tmp_test.append(label_index[class_type][i])
        train_node.extend(tmp_train)
        val_node.extend(tmp_val)
        test_node.extend(tmp_test)

    random.shuffle(train_node)
    return train_node, val_node, test_node

# 获取图set_list表示{0:[邻居], 1:[], 2:[], .....} 1
def get_adj_dict(adj):
    adj_dict = {}
    tmp = np.nonzero(adj)
    for i in range(len(tmp[0])):
        if tmp[0][i] not in adj_dict.keys():
            adj_dict[tmp[0][i]] = []
        adj_dict[tmp[0][i]].append(tmp[1][i])

    return adj_dict

def get_k_step_neighbor(node, k, adj_dict, addition_node=None):
    result = [node]
    neighbor = [node]
    now_k = 0
    while True:
        tmp = set()
        for check_node in neighbor:
            check_node_neighbor = adj_dict[check_node]
            for check_node2 in check_node_neighbor:
                if check_node2 not in result:
                    tmp.add(check_node2)
        tmp = list(tmp)
        if len(tmp) > 0:
            neighbor = tmp
            result.extend(tmp)
        else:
            break
        now_k += 1
        if now_k >= k:
            break
    if addition_node is not None:
        for t_node in addition_node:
            if t_node not in result:
                result.append(t_node)
    return result

# 获取一个节点的k阶邻居{0:[], 1:[], 2:[], ....., k:[]}
def get_k_step_neighbor2(adj_dict, node, k):
    neighbor = {}
    neighbor[0] = [node]
    sampled_node = [node]
    now_k = 1
    while now_k <= k:
        tmp = set()
        for check_node in neighbor[now_k-1]:
            check_node_neighbor = adj_dict[check_node]
            for check_node2 in check_node_neighbor:
                if check_node2 not in sampled_node:
                    tmp.add(check_node2)
                    sampled_node.append(check_node2)
        tmp = list(tmp)
        if len(tmp) > 0:
            neighbor[now_k] = tmp
            now_k += 1
        else:
            break
    return neighbor

class TrainUtil(object):
    def __init__(self, train_node=None, val_node=None, test_node=None, labels=None, batch_size=64, **kwargs):
        super(TrainUtil, self).__init__(**kwargs)
        self.train_node = train_node
        self.val_node = val_node
        self.test_node = test_node
        self.labels = labels
        self.batch_size = batch_size
        self.batch_num = 0

    def set_train_node(self, train_node):
        self.train_node = train_node

    def set_val_node(self, val_node):
        self.val_node = val_node

    def set_test_node(self, test_node):
        self.test_node = test_node

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def shuffle(self, type="train"):
        if type == "train":
            self.train_node = np.random.permutation(self.train_node)
        elif type == "val":
            self.val_node = np.random.permutation(self.val_node)
        elif type == "test":
            self.test_node = np.random.permutation(self.test_node)

    def reset(self):
        self.batch_num = 0
        self.shuffle("train")

    def next_minibatch_feed_dict(self, type="train", new_batch=None):
        if new_batch is None:
            batch_size = self.batch_size
        else:
            batch_size = new_batch
        start_idx = self.batch_num * batch_size
        self.batch_num += 1
        if type == "train":
            end_idx = min(start_idx + batch_size, len(self.train_node))
            batch_nodes = self.train_node[start_idx: end_idx]
            return batch_nodes, self.labels[batch_nodes]
        elif type == "test":
            end_idx = min(start_idx + batch_size, len(self.test_node))
            batch_nodes = self.test_node[start_idx: end_idx]
            return batch_nodes, self.labels[batch_nodes]
        elif type == "val":
            end_idx = min(start_idx + batch_size, len(self.val_node))
            batch_nodes = self.val_node[start_idx: end_idx]
            return batch_nodes, self.labels[batch_nodes]

    def end(self, type="train", new_batch=None):
        if new_batch is None:
            batch_size = self.batch_size
        else:
            batch_size = new_batch
        if type == "train":
            return self.batch_num * batch_size >= len(self.train_node)
        elif type == "test":
            return self.batch_num * batch_size >= len(self.test_node)
        elif type == "val":
            return self.batch_num * batch_size >= len(self.val_node)

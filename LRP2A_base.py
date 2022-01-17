# -*- coding:utf-8 -*-


import tensorflow as tf
from utils import TrainHelp
import numpy as np
from utils.data_loader import DataLoader



class SurrogateBase(tf.keras.Model):
    def __init__(self, feature_dim, class_num, nhid, dtype=tf.float32, **kwargs):
        super(SurrogateBase, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.class_num = class_num
        self.nhid = nhid
        self.var_dtype = dtype
        self.__dict__["convolution_layer"] = []
        self.build(self.nhid)

    def build(self, input_shape):
        ini = tf.keras.initializers.glorot_uniform()
        input_dim = self.feature_dim
        for i in range(len(input_shape)):
            tmp = tf.Variable(ini(shape=[input_dim, input_shape[i]], dtype=self.var_dtype), trainable=True)
            setattr(self, "layer_" + str(i), tmp)
            input_dim = input_shape[i]
            self.convolution_layer.append(getattr(self, "layer_" + str(i)))
        tmp = tf.Variable(ini(shape=[input_dim, self.class_num], dtype=self.var_dtype), trainable=True)
        setattr(self, "layer_" + str(i + 1), tmp)
        self.convolution_layer.append(getattr(self, "layer_" + str(i + 1)))
        super(SurrogateBase, self).build([])

    def call(self, adj, feature):
        L = tf.Variable(tf.matmul(adj, adj), dtype=self.var_dtype, trainable=False)
        R = self.convolution_layer[0]
        for layer in self.convolution_layer[1:]:
            R = tf.matmul(R, layer)
        out = tf.matmul(tf.matmul(L, feature), R)
        return tf.keras.activations.softmax(out)


class SurrogateGNN(object):
    def __init__(self, adj, feature, class_num, nhid, dtype=tf.float32, adj_dict=None):
        self.surrogate_model = SurrogateBase(feature.shape[1], class_num, nhid)
        if adj_dict is not None:
            self.adj_dict = adj_dict
        else:
            self.adj_dict = TrainHelp.get_adj_dict(adj)
        self.feature = feature
        self.nhid = nhid
        self.var_dtype = dtype

    def get_predict(self, node, adj_type="self"):
        data_loader = DataLoader(self.adj_dict, self.feature, labels=None, train_node=node, nhid=len(self.nhid) + 2,
                                     batch_size=32, shuffle=False, thread_num=8, adj_type=adj_type)
        now_batch = 0
        pre_node = []
        pre_index = []
        pre_prop = []

        for (new_adj, new_feature, new_label, new_node) in iter(data_loader):
            result = self.surrogate_model.call(new_adj, new_feature)
            need = [i for i in range(new_label.shape[0])]
            result = tf.gather(result, need)
            logi = tf.argmax(result, axis=1)
            pre_index.extend(logi.numpy().tolist())
            pre_node.extend(new_node)
            result = result.numpy()
            for i in need:
                pre_prop.append(result[i][logi[i]])
            now_batch += 1
            if now_batch%4 == 0:
                print("\r", now_batch, end="")
        data_loader.end_process()
        print()
        return pre_index, pre_prop, pre_node

    def train(self, labels, Epoch=40, train_node=None, val_node=None, batch_size=32, split_rate=0.1, show=True,
               update=True, save=None, adj_type="self", thread_num=8):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        ca_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        if train_node is None:
            train_node, val_node = TrainHelp.get_train_val_node(labels, spilt_rate=split_rate)

        train_data_loader = DataLoader(self.adj_dict, self.feature, labels, train_node, nhid=len(self.nhid) + 2,
                                       batch_size=batch_size, shuffle=True, thread_num=thread_num, adj_type=adj_type)
        val_data_loader = DataLoader(self.adj_dict, self.feature, labels, val_node, nhid=len(self.nhid) + 2,
                                     batch_size=batch_size, shuffle=True, thread_num=thread_num, adj_type=adj_type)
        max_acc = 0.0
        for epoch in range(Epoch):
            epoch_train_loss = tf.keras.metrics.Mean()
            epoch_train_acc = tf.keras.metrics.CategoricalAccuracy()
            now_bath = 0
            for (new_adj, new_feature, new_label, _) in iter(train_data_loader):
                with tf.GradientTape() as tape:
                    result = self.surrogate_model.call(new_adj, new_feature)
                    need = [i for i in range(new_label.shape[0])]
                    result = tf.gather(result, need)
                    train_loss = ca_loss_fn(y_true=new_label, y_pred=result)
                if update:
                    grads = tape.gradient(train_loss, self.surrogate_model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.surrogate_model.trainable_variables))

                epoch_train_loss(train_loss)
                epoch_train_acc(new_label, result)
                if show:
                    print("\r Epoch:{:3d} train_iter:{:4d} train_acc={:.6f} train_loss:{:.6f} ".format(epoch, now_bath,
                                                                                                       epoch_train_acc.result().numpy(),
                                                                                                       epoch_train_loss.result().numpy()), end="")
                now_bath += 1
            if show:
                print()

            now_bath = 0
            epoch_val_loss = tf.keras.metrics.Mean()
            epoch_val_acc = tf.keras.metrics.CategoricalAccuracy()

            for (new_adj, new_feature, new_label, _) in iter(val_data_loader):
                result = self.surrogate_model.call(new_adj, new_feature)
                need = [i for i in range(new_label.shape[0])]
                result = tf.gather(result, need)
                loss = ca_loss_fn(y_true=new_label, y_pred=result)
                epoch_val_loss(loss)
                epoch_val_acc(new_label, result)
                if show:
                    print("\r             val_iter:{:4d}   val_acc={:.6f}   val_loss:{:.6f}".format(now_bath,
                                                                                                    epoch_val_acc.result().numpy(),
                                                                                                    epoch_val_loss.result().numpy()), end="")
                now_bath += 1
            if show:
                print()
            if epoch_val_acc.result().numpy() > max_acc:
                max_acc = epoch_val_acc.result().numpy()
                acc_no_up = 0
                if save is not None:
                    self.surrogate_model.save_weights(save)
            else:
                acc_no_up += 1
            if acc_no_up > 10:
                break
        train_data_loader.end_process()
        val_data_loader.end_process()
        return max_acc

    # 我自己的聚合方式
    def _calculate_A1(self):
        A_ = np.zeros((len(self.adj_dict), len(self.adj_dict)) , dtype=np.float32)
        all_count =len(self.adj_dict)
        now_count = 0
        for target_node in self.adj_dict.keys():  # 遍历某个连通图中的点
            t_count = 1 + len(self.adj_dict[target_node]) + 1
            rate = 1 / t_count
            t_rate = 1 / t_count / (all_count-t_count+1)
            A_[target_node, :] = t_rate
            A_[target_node, target_node] = rate
            A_[target_node, self.adj_dict[target_node]] = rate
            now_count += 1
        return A_

    def train2(self, labels, Epoch=40, train_node=None, val_node=None, show=True, update=True, save=None):
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        ca_loss_fn = tf.keras.losses.CategoricalCrossentropy()
        feature = self.feature
        A_ = self._calculate_A1()
        true_index = np.argmax(labels, axis=1)

        max_acc = 0.0
        acc_no_up = 0
        for iter in range(Epoch):
            with tf.GradientTape() as tape:
                result = self.surrogate_model.call(A_, feature)
                train_result = tf.gather(result, train_node)
                train_loss = ca_loss_fn(y_true=labels[train_node], y_pred=train_result)
            if update:
                grads = tape.gradient(train_loss, self.surrogate_model.trainable_variables)
                optimizer.apply_gradients(zip(grads, self.surrogate_model.trainable_variables))
            val_result = tf.gather(result, val_node)
            val_loss = ca_loss_fn(y_true=labels[val_node], y_pred=val_result)
            y_pre = tf.argmax(result, axis=1)
            train_pre = tf.gather(y_pre, train_node)
            val_pre = tf.gather(y_pre, val_node)
            train_true = true_index[train_node]
            val_true = true_index[val_node]
            if show:
                print("iter{:3d}  train_acc: {:.6f} loss: {:.6f} val_acc: {:.6f} loss: {:.6f}".format(iter, np.mean(
                    train_pre == train_true), train_loss, np.mean(val_pre == val_true), val_loss))
            if np.mean(val_pre == val_true) > max_acc:
                max_acc = np.mean(val_pre == val_true)
                acc_no_up = 0
                if save is not None:
                    self.surrogate_model.save_weights(save)
            else:
                acc_no_up += 1
            if acc_no_up > 30:
                break
        return max_acc

class LRP2A(SurrogateGNN):
    # 计算扰动需要用到的
    def compute_alpha(self, n, S_d, d_min):
        # n 大于d_min的个数  S_d log(度>d_min)和 d_min 2
        return n / (S_d - n * np.log(d_min - 0.5)) + 1

    # 计算扰动需要用到的
    def compute_log_likelihood(self, n, alpha, S_d, d_min):
        # n 大于d_min的个数  S_d log(度>d_min)和 d_min 2 alpha上一步算的
        return n * np.log(alpha) + n * alpha * np.log(d_min) + (alpha + 1) * S_d

    # 扰动初始相关值
    def get_base_value_raodong(self, degree_ori, d_min):
        log_d_ori = np.sum(np.log(degree_ori[degree_ori >= d_min]))
        n_ori = np.sum(degree_ori >= d_min)
        alpha_ori = self.compute_alpha(n_ori, log_d_ori, d_min)  # 公式6
        log_likelihood_ori = self.compute_log_likelihood(n_ori, alpha_ori, log_d_ori, d_min)  # 公式7
        return log_d_ori, n_ori, log_likelihood_ori

    # 计算修改后的扰动大小
    def get_after_vale_raodong(self, degree_new, d_min, log_d_ori, n_ori, log_likelihood_ori):
        log_d_new = np.sum(np.log(degree_new[degree_new >= d_min]))
        n_new = np.sum(degree_new >= d_min)
        alpha_new = self.compute_alpha(n_new, log_d_new, d_min)  # 公式6
        log_likelihood_new = self.compute_log_likelihood(n_new, alpha_new, log_d_new, d_min)  # 公式7
        alphas_combined_new = self.compute_alpha(n_new + n_ori, log_d_new + log_d_ori, d_min)
        new_ll_combined_new = self.compute_log_likelihood(n_new + n_ori, alphas_combined_new, log_d_new + log_d_ori,
                                                          d_min)
        ratios_new = -2 * new_ll_combined_new + 2 * (log_likelihood_new + log_likelihood_ori)
        return ratios_new





    def calculate_A1(self, adj=None, adj_dict=None, show=False):
        if adj_dict is None:
            adj_dict = TrainHelp.get_adj_dict(adj)
        all_count = len(adj_dict)
        A_ = np.zeros((all_count, all_count), dtype=np.float32)

        now_count = 0
        for target_node in range(all_count):  # 遍历某个连通图中的点
            t_count = 1 + len(adj_dict[target_node]) + 1
            rate = 1 / t_count
            t_rate = 0.0
            if all_count > (1 + len(adj_dict[target_node])):
                t_rate = rate / (all_count - 1 - len(adj_dict[target_node]))
            A_[target_node, :] = t_rate
            A_[target_node, target_node] = rate
            A_[target_node, adj_dict[target_node]] = rate
            now_count += 1
            if show:
                print("\r calculate_A1 {:.2f}".format(now_count / all_count), end="")
        return A_

    # 攻击时前向计算一次
    def forward_option(self, A_, feature):
        self.tmp_a = tf.constant(A_, dtype=self.var_dtype)
        # self.tmp_a = tmp_a / tf.expand_dims(tf.reduce_sum(tmp_a, axis=1), 1)
        self.L = tf.matmul(self.tmp_a, self.tmp_a)
        self.C = tf.matmul(self.L, feature)
        self.out = tf.matmul(self.C, self.R)

    # 获取将要改变成的类和现在预测的类
    def get_second_pre(self, node, true_index, target_ca=None):
        pre_label_index = tf.argsort(self.out, direction="DESCENDING", axis=1).numpy()

        second = pre_label_index[node][0]
        if pre_label_index[node][0] == true_index:
            second = pre_label_index[node][1]
        if target_ca is not None:
            second = target_ca
        return second, pre_label_index[node][0]

    # lrp计算过程
    def lrp_option(self, node, label, feature):
        rate1 = tf.Variable([[1 / self.out[node][label]]], dtype=tf.float32)
        node_C = self.C[node]
        node_out = tf.transpose(self.R)[label]
        rate2 = tf.matmul(rate1, [node_C * node_out])
        node_L = self.L[node]
        node_tmp_feature = tf.transpose(feature)
        rate2_t = rate2 / (self.C[node] + 0.000000000001)
        rate3 = tf.matmul(rate2_t, node_L * node_tmp_feature)
        rate3_t = rate3 / (self.L[node] + 0.000000000001)
        node_tmp_a = tf.expand_dims(tf.transpose(self.tmp_a)[:, node], 1)
        out1 = node_tmp_a * self.tmp_a
        out2 = rate3_t * out1
        return out2

    def change_A(self, A_, last, adj_dict):
        if last[0] == -1:
            return
        for target_node in last:
            t_count = 1 + len(adj_dict[target_node]) + 1
            rate = 1 / t_count
            t_rate = 0.0
            if len(adj_dict) > 1 + len(adj_dict[target_node]):
                t_rate = rate / (len(adj_dict) - 1 - len(adj_dict[target_node]))
            A_[target_node, :] = t_rate
            A_[target_node, target_node] = rate
            A_[target_node, adj_dict[target_node]] = rate

    # 计算修改后的影响
    def calculate_score(self, new_adj_dict, edge, node, true_index, feature, target_ca=None, A_=None):
        t0, t1 = edge[0], edge[1]
        if t0 in new_adj_dict[t1]:
            new_adj_dict[t0].remove(t1)
            new_adj_dict[t1].remove(t0)
        else:
            new_adj_dict[t0].append(t1)
            new_adj_dict[t1].append(t0)
        # A_ = self.calculate_A1(adj_dict=new_adj_dict)
        self.change_A(A_, edge, new_adj_dict)
        self.forward_option(A_, feature)
        second, _ = self.get_second_pre(node, true_index, target_ca=target_ca)
        tmp = tf.keras.activations.softmax(self.out)
        score = tmp[node][true_index] - tmp[node][second]
        if t0 in new_adj_dict[t1]:
            new_adj_dict[t0].remove(t1)
            new_adj_dict[t1].remove(t0)
        else:
            new_adj_dict[t0].append(t1)
            new_adj_dict[t1].append(t0)
        self.change_A(A_, edge, new_adj_dict)
        return score

    def get_perturate_edge(self, important_score, action, last, last_action, node, per_node, repe_flag, repe_edge, new_adj_dict, direction=True, true_index=0, degree=None, d_min=None, log_d_ori=None, n_ori=None, log_likelihood_ori=None):
        PER = None
        node_count = len(new_adj_dict.keys())

        if direction is True: ## 直接攻击
            sort_index = tf.argsort(important_score, direction="DESCENDING")
            i = node
            for s_j in range(node_count):
                j = sort_index[i][s_j].numpy()

                # action 1添加  0删除   本来就连着或本来就没连
                if j == i or (action == 0 and j not in new_adj_dict[i]) or (action == 1 and j in new_adj_dict[i]):
                    continue
                # 上一次连的就是这条边，陷入循环
                if repe_flag is True and last_action == int(1-action) and (
                        (last[0] == per_node[i] and last[1] == per_node[j]) or (
                        last[1] == per_node[i] and last[0] == per_node[j])):
                    continue
                if (per_node[i], per_node[j]) in repe_edge:
                    continue

                if action == 0:
                    # 删掉后连通子图增多，出现孤立点
                    new_adj_dict[i].remove(j)
                    new_adj_dict[j].remove(i)
                    if len(new_adj_dict[node]) == 0:
                        new_adj_dict[i].append(j)
                        new_adj_dict[j].append(i)
                        continue
                    new_adj_dict[i].append(j)
                    new_adj_dict[j].append(i)
                if action == 0:
                    degree[per_node[i]] -= 1
                    degree[per_node[j]] -= 1
                    ratios_new = self.get_after_vale_raodong(degree, d_min, log_d_ori, n_ori, log_likelihood_ori)
                    degree[per_node[i]] += 1
                    degree[per_node[j]] += 1
                    if ratios_new >= 0.004:
                        continue
                elif action == 1:
                    degree[per_node[i]] += 1
                    degree[per_node[j]] += 1
                    ratios_new = self.get_after_vale_raodong(degree, d_min, log_d_ori, n_ori, log_likelihood_ori)
                    degree[per_node[i]] -= 1
                    degree[per_node[j]] -= 1
                    if ratios_new >= 0.004:
                        continue
                PER = (i, j)
                break
        elif direction is False:  # 间接攻击
            sss = tf.reshape(tf.gather(important_score, new_adj_dict[node]), [-1]).numpy()
            sort_index = tf.argsort(sss, direction="DESCENDING")
            count = 0
            now = 0

            while True:
                if count > 5000:
                    break
                i = node
                j = node

                while (i == node or j == node) and now < len(sort_index):
                    m = sort_index[now]
                    now += 1
                    i, j = (m // node_count).numpy(), (m % node_count).numpy()

                    i = new_adj_dict[node][i]
                    if j < i or i == j or (action == 0 and j not in new_adj_dict[i]) or (action == 1 and j in new_adj_dict[i]):
                        i = node
                        j = node
                count += 1

                if i == node or j == node:
                    continue

                # 上一次连的就是这条边，陷入循环
                if repe_flag is True and last_action == int(1-action) and (
                        (last[0] == per_node[i] and last[1] == per_node[j]) or (
                        last[1] == per_node[i] and last[0] == per_node[j])):
                    continue
                if (per_node[i], per_node[j]) in repe_edge:
                    continue


                if action == 0:
                    # 删掉后连通子图增多，出现孤立点
                    new_adj_dict[i].remove(j)
                    new_adj_dict[j].remove(i)
                    if len(new_adj_dict[node]) == 0:
                        new_adj_dict[i].append(j)
                        new_adj_dict[j].append(i)
                        continue
                    new_adj_dict[i].append(j)
                    new_adj_dict[j].append(i)

                if action == 0:
                    degree[per_node[i]] -= 1
                    degree[per_node[j]] -= 1
                    ratios_new = self.get_after_vale_raodong(degree, d_min, log_d_ori, n_ori, log_likelihood_ori)
                    degree[per_node[i]] += 1
                    degree[per_node[j]] += 1
                    if ratios_new >= 0.004:
                        continue
                elif action == 1:
                    degree[per_node[i]] += 1
                    degree[per_node[j]] += 1
                    ratios_new = self.get_after_vale_raodong(degree, d_min, log_d_ori, n_ori, log_likelihood_ori)
                    degree[per_node[i]] -= 1
                    degree[per_node[j]] -= 1
                    if ratios_new >= 0.004:
                        continue

                PER = (i, j)
                break

        return PER

    def attack_init(self):
        return None

    def get_lrp_value(self, ori_adj, ori_feature, node, ori_path, target_ca=None):
        self.load_weights(ori_path)  # 加载原始模型
        self.feature = tf.constant(ori_feature, dtype=self.var_dtype)
        adj_dict = TrainHelp.get_adj_dict(ori_adj)
        A_ = self.calculate_A1(ori_adj, adj_dict)
        R = self.convolution_layer[0]
        for layer in self.convolution_layer[1:]:
            R = tf.matmul(R, layer)
        self.R = R
        self.forward_option(A_)
        predict = self.out[node]
        print(predict)

        true_index = tf.argmax(predict).numpy()
        second, _ = self.get_second_pre(node, true_index, target_ca=target_ca)
        # true_index = 1
        lrp_value = self.lrp_option(node, true_index)
        lrp_value = lrp_value + tf.transpose(lrp_value)
        return lrp_value, true_index


    def watch(self, node, ori_adj, ori_feature, ori_labels, ori_path):
        self.surrogate_model.load_weights(ori_path)  # 加载原始模型
        self.feature = tf.constant(ori_feature, dtype=self.var_dtype)
        adj_dict = TrainHelp.get_adj_dict(ori_adj)
        A_ = self.calculate_A1(ori_adj, adj_dict)
        R = self.surrogate_model.convolution_layer[0]
        for layer in self.surrogate_model.convolution_layer[1:]:
            R = tf.matmul(R, layer)
        self.R = R
        self.forward_option(A_, self.feature)
        predict = self.out[node]
        print(predict)

        true_index = tf.argmax(predict).numpy()
        second, _ = self.get_second_pre(node, true_index, target_ca=None)

        true_important_score = self.lrp_option(node, true_index, ori_feature)
        true_important_score = true_important_score + tf.transpose(true_important_score)  # 对称
        second_important_score = self.lrp_option(node, second, ori_feature)
        second_important_score = second_important_score + tf.transpose(second_important_score)
        return true_important_score, second_important_score, true_index, second


    def attack(self, t_node, victim_label_index, per=10, show=False, iter=100, ori_path=None, target_ca=None, up_to_success=0, direct=True, all_graph=False, raodong=True, d_min=2):
        ACTION_FLAG = {"增加":1, "删除":0}
        self.surrogate_model.load_weights(ori_path)  # 加载原始模型

        change_list = []  # 修改的列表

        last = (-1, -1)  # 这是上一次修改的
        last_action = -1  # 动作 1添加 0删除
        repe_flag = False  # 是否要注意
        repe_edge = []

        #  模型参数乘
        R = self.surrogate_model.convolution_layer[0]
        for layer in self.surrogate_model.convolution_layer[1:]:
            R = tf.matmul(R, layer)
        self.R = R

        tmp_adj_dict = {}

        for key in self.adj_dict.keys():
            tmp_adj_dict[key] = []
            for val in self.adj_dict[key]:
                tmp_adj_dict[key].append(val)

        true_index = victim_label_index[t_node]


        new_adj_dict = tmp_adj_dict
        new_feature = self.feature
        per_node = [i for i in range(len(new_adj_dict.keys()))]
        node = t_node
        A_ = self.calculate_A1(adj_dict=new_adj_dict)

        if raodong:
            ori_degree = [len(new_adj_dict[i]) for i in range(len(new_adj_dict.keys()))]
            ori_degree = np.array(ori_degree)
            log_d_ori, n_ori, log_likelihood_ori = self.get_base_value_raodong(ori_degree, d_min)

        while (up_to_success == 0 and len(change_list) < per and iter > 0) or (up_to_success == 1 and iter > 0):
            iter -= 1


            self.change_A(A_, last, new_adj_dict)

            # 进行一次预测过程
            self.forward_option(A_, new_feature)
            # 得到第二可能的
            second_pre, max_pre = self.get_second_pre(node, true_index, target_ca=target_ca)


            # 获取预测为不同类时，每条边的重要度
            true_important_score = self.lrp_option(node, true_index, new_feature)
            true_important_score = true_important_score + tf.transpose(true_important_score)  # 对称
            second_important_score = self.lrp_option(node, second_pre, new_feature)
            second_important_score = second_important_score + tf.transpose(second_important_score)


            ADD = None
            DEL = None

            # 选择边
            ADD = self.get_perturate_edge(important_score=second_important_score, action=ACTION_FLAG["增加"], last=last, last_action=last_action, node=node, per_node=per_node,
                                          repe_flag=repe_flag, repe_edge=repe_edge, new_adj_dict=new_adj_dict, direction=direct, true_index=true_index,
                                          degree=ori_degree, d_min=d_min, log_d_ori=log_d_ori, n_ori=n_ori, log_likelihood_ori=log_likelihood_ori)
            DEL = self.get_perturate_edge(important_score=true_important_score, action=ACTION_FLAG["删除"], last=last, last_action=last_action, node=node, per_node=per_node,
                                          repe_flag=repe_flag, repe_edge=repe_edge, new_adj_dict=new_adj_dict, direction=direct,
                                          degree=ori_degree, d_min=d_min, log_d_ori=log_d_ori, n_ori=n_ori, log_likelihood_ori=log_likelihood_ori)


            # 计算每种方式的得分 越小越好

            # 计算得分
            score_add, score_del = 10000000, 10000000
            score_flag = False
            if ADD is not None:
                score_flag = True
                score_add = self.calculate_score(new_adj_dict, edge=ADD, node=node, true_index=true_index, feature=new_feature, target_ca=target_ca, A_=A_)
            if DEL is not None:
                score_flag = True
                score_del = self.calculate_score(new_adj_dict, edge=DEL, node=node, true_index=true_index, feature=new_feature, target_ca=target_ca, A_=A_)

            repe_flag = False
            if score_flag:
                if score_add < score_del:
                    now_opt = 1
                    now_info = "增加"
                    t0 = per_node[ADD[0]]
                    t1 = per_node[ADD[1]]
                    tmp_adj_dict[t0].append(t1)
                    tmp_adj_dict[t1].append(t0)
                    ori_degree[t0] += 1
                    ori_degree[t1] += 1

                else:
                    now_opt = 0
                    now_info = "删除"
                    t0 = per_node[DEL[0]]
                    t1 = per_node[DEL[1]]
                    tmp_adj_dict[t0].remove(t1)
                    tmp_adj_dict[t1].remove(t0)
                    ori_degree[t0] -= 1
                    ori_degree[t1] -= 1

                # 如果上次是删边并且删的这条/上次是增边并且增的这条边
                if last_action == int(1-now_opt) and (
                        (last[0] == t0 and last[1] == t1) or (last[0] == t1 and last[1] == t0)):
                    repe_flag = True
                    repe_edge.append((t0, t1))
                    repe_edge.append((t1, t0))
                if len(change_list) > 0 and ((change_list[-1][0] == t0 and change_list[-1][1] == t1) or (
                        change_list[-1][0] == t1 and change_list[-1][1] == t0)):
                    repe_flag = True
                    repe_edge.append((t0, t1))
                    repe_edge.append((t1, t0))
                last = (t0, t1)
                last_action = now_opt
                if last in change_list:
                    change_list.remove(last)
                else:
                    change_list.append(last)

                if show:
                    print("{:s}边({:d},{:d})".format(now_info, t0, t1))


                if up_to_success == 1 and (score_del < 0 or score_add < 0):
                    break

            else:
                break
        if show:
            print("修改了：", len(change_list))
        return change_list








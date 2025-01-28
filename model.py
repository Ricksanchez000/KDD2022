from __future__ import absolute_import, division, print_function

from collections import namedtuple, defaultdict

import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch
import numpy as np
import math

from replay import Replay
from utils.logger import info
from utils.tools import feature_state_generation, justify_operation_type, insert_generated_feature_to_original_feas

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

O1 = ['sqrt', 'square', 'sin', 'cos', 'tanh', 'stand_scaler',
      'minmax_scaler', 'quan_trans', 'sigmoid', 'log', 'reciprocal']
O2 = ['+', '-', '*', '/']
O3 = ['stand_scaler', 'minmax_scaler', 'quan_trans']
operation_set = O1 + O2
one_hot_op = pd.get_dummies(operation_set)
operation_emb = defaultdict()
for item in one_hot_op.columns:
    operation_emb[item] = torch.tensor(one_hot_op[item].values, dtype=torch.float32)

OP_DIM = len(operation_set)


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ClusterNet(nn.Module):

    def __init__(self, STATE_DIM, ACTION_DIM, HIDDEN_DIM=100, init_w=0.1):
        super(ClusterNet, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_DIM)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.out = nn.Linear(HIDDEN_DIM, 1)
        self.out.weight.data.normal_(-init_w, init_w)

        #对于Cluster2来说，hidden_dim = (STATE_DIM + OP_DIM) * 2 = (64 + 15)*2 = 158
        #同时，输入的STATE_DIM = STATE_DIM + OP_DIM = 64 + 15 = 79
        #ACTION_DIM保持不变为 64 ， 则fc1初始化时，input_dim = 79 + 64 = 143
        #但是，我们当前输入的x仍然是 64 + 64 = 128
                
    def forward(self, x):
        x = self.fc1(x)     #这里有问题，输入x的维度是8*68， 但是fc1layer是128*128 为啥呢？先看看X是什么
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, cluster_state_dim, hidden_dim, gamma, device, memory: Replay,
                 ent_weight, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, init_w=1e-6):
        super(DQNNetwork, self).__init__()
        self.state_dim = state_dim
        self.cluster_state_dim = cluster_state_dim
        self.hidden_dim = hidden_dim
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.GAMMA = gamma
        self.ENT_WEIGHT = ent_weight
        self.cuda_info = device is not None
        self.operation_emb = dict()
        self.memory = memory
        self.learn_step_counter = 0
        self.init_w = init_w
        self.TARGET_REPLACE_ITER = 100
        self.loss_func = nn.MSELoss()
        self.BATCH_SIZE = memory.BATCH_SIZE #后续需要用到batch_size，仅在memory中被定义，需要取出
        for k, v in operation_emb.items():
            if self.cuda_info:
                v = v.cuda()
            self.operation_emb[k] = v

    @staticmethod
    def _operate_two_features(f_cluster1, f_cluster2, op, f_names1, f_names2):
        assert op in O2
        if op == '/' and np.sum(f_cluster2 == 0) > 0:
            return None, None
        op_func = justify_operation_type(op)
        feas, feas_names = [], []
        for i in range(f_cluster1.shape[1]):
            for j in range(f_cluster2.shape[1]):
                feas.append(op_func(f_cluster1[:, i], f_cluster2[:, j]))
                feas_names.append(str(f_names1[i]) + op + str(f_names2[j]))
        feas = np.array(feas)
        feas_names = np.array(feas_names)
        return feas.T, feas_names

    @staticmethod
    def _operate_one_feature(f_cluster1, f_names1, op):
        assert op in O1 or op in O3
        feas = None
        feas_name = None
        op_sign = justify_operation_type(op)
        f_new, f_new_name = [], []
        if op == 'sqrt':
            for i in range(f_cluster1.shape[1]):
                if np.sum(f_cluster1[:, i] < 0) == 0:
                    f_new.append(op_sign(f_cluster1[:, i]))
                    f_new_name.append(str(f_names1[i]) + '_' + str(op))
            f_generate = np.array(f_new).T
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'reciprocal':
            for i in range(f_cluster1.shape[1]):
                if np.sum(f_cluster1[:, i] == 0) == 0:
                    f_new.append(op_sign(f_cluster1[:, i]))
                    f_new_name.append(str(f_names1[i]) + '_' + str(op))
            f_generate = np.array(f_new).T
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op == 'log':
            for i in range(f_cluster1.shape[1]):
                if np.sum(f_cluster1[:, i] <= 0) == 0:
                    f_new.append(op_sign(f_cluster1[:, i]))
                    f_new_name.append(str(f_names1[i]) + '_' + str(op))
            f_generate = np.array(f_new).T
            if len(f_generate) != 0:
                feas = f_generate
                feas_name = f_new_name
        elif op in O3:
            feas = op_sign.fit_transform(f_cluster1)
            feas_name = [(str(f_n) + '_' + str(op)) for f_n in f_names1]
        else:
            feas = op_sign(f_cluster1)
            feas_name = [(str(f_n) + '_' + str(op)) for f_n in f_names1]
        return feas, feas_name

    def op(self, data, f_cluster1, f_names1, op, f_cluster2=None, f_names2=None):
        if op in O2:
            assert f_cluster2 is not None and f_names2 is not None
            f_generate, final_name = self._operate_two_features(f_cluster1, f_cluster2, op, f_names1, f_names2)
        else:
            f_generate, final_name = self._operate_one_feature(f_cluster1, f_names1, op)
        is_op = False
        if f_generate is None or final_name is None:
            return data, is_op
        f_generate = pd.DataFrame(f_generate)
        f_generate.columns = final_name
        public_name = np.intersect1d(np.array(data.columns), final_name)
        if len(public_name) > 0:
            reduns = np.setxor1d(final_name, public_name)
            if len(reduns) > 0:
                is_op = True
                f_generate = f_generate[reduns]
                Dg = insert_generated_feature_to_original_feas(data, f_generate)
            else:
                Dg = data
        else:
            is_op = True
            Dg = insert_generated_feature_to_original_feas(data, f_generate)
        return Dg, is_op

    def learn(self, optimizer):
        raise NotImplementedError()


class ClusterDQNNetwork(DQNNetwork):
    def __init__(self, state_dim, cluster_state_dim, hidden_dim, memory: Replay, ent_weight, select='head', gamma=0.99
                 , EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200,
                 device=None, init_w=1e-6):
        super(ClusterDQNNetwork, self).__init__(state_dim, cluster_state_dim, hidden_dim, gamma, device,
                                                 memory, ent_weight, EPS_START=EPS_START,
                                                 EPS_END=EPS_END, EPS_DECAY=EPS_DECAY, init_w=init_w)
        assert select in {'head', 'tail'}
        self.select_mode = select == 'head'
        self.eval_net, self.target_net = ClusterNet(self.state_dim, self.cluster_state_dim, self.hidden_dim, init_w=self.init_w), ClusterNet(
            self.state_dim, self.cluster_state_dim, self.hidden_dim, init_w=self.init_w)

    def get_q_value(self, state_emb, action):
        return self.eval_net(torch.cat((state_emb, action)))

    def get_q_value_next(self, state_emb, action):
        return self.target_net(torch.cat((state_emb, action)))

    def get_op_emb(self, op):
        if type(op) is int:
            assert op >= 0 and op < len(operation_set)
            return self.operation_emb[operation_set[op]]
        elif type(op) is str:
            assert op in operation_set
            return self.operation_emb[op]
        else:  # is embedding
            return op
    #Q(state,cluster) 所以DIM是 64 + 64
    #Q(s,a)这里的a对应的dim即x中提取的cluster子集的REP = 64
    def forward(self, clusters=None, X=None, cached_state_emb=None, cached_cluster_state=None, for_next=False):
        if cached_state_emb is None:
            assert clusters is not None
            assert X is not None
            assert self.select_mode
            state_emb = feature_state_generation(pd.DataFrame(X)) #这里构建整个group的展开两次的REP，
            state_emb = torch.FloatTensor(state_emb)              #然后转化为张量,应该是8*8 = 64  
            if self.cuda_info:
                state_emb = state_emb.cuda()
        else:
            state_emb = cached_state_emb
        q_vals, cluster_list, select_cluster_state_list = [], [], dict()
        if clusters is None:
            iter = cached_cluster_state.items()
        else:
            iter = clusters.items()
        #遍历clusters来计算对应的Q(state,cluster)
        #传入计算q值的维度就是 64 + 64 = 128 = (len(state_emb) + len(select_cluster_state))
        for cluster_index, value in iter:
            if clusters is None:
                select_cluster_state = value
            else:
                assert X is not None
                select_cluster_state = feature_state_generation(pd.DataFrame(X[:, list(value)]))
                select_cluster_state = torch.FloatTensor(select_cluster_state)
                if self.cuda_info:
                    select_cluster_state = select_cluster_state.cuda()
            select_cluster_state_list[cluster_index] = select_cluster_state
            if for_next:
                q_val = self.get_q_value_next(state_emb, select_cluster_state)
                print(f"Shape of state_emb: {state_emb.shape}; cluster_index {cluster_index}; shape of select_cluster_state: {select_cluster_state.shape}")
            else:
                q_val = self.get_q_value(state_emb, select_cluster_state)#它最后是把这两个concate在了一起，然后送入网络中，select_cluster_state成为了ACTION_DIM
                print(f"Shape of state_emb: {state_emb.shape}; cluster_index {cluster_index}; shape of select_cluster_state: {select_cluster_state.shape}") 
            q_vals.append(q_val.detach())  # th
            cluster_list.append(cluster_index)
        q_vals_ = [None] * len(q_vals)
        for index, pos in enumerate(cluster_list):
            q_vals_[pos] = q_vals[index]
        return q_vals_, select_cluster_state_list, state_emb

    def store_transition(self, s1, a1, r, s2, a2):
        self.memory.store_transition((s1, a1, r, s2, a2))  #在这里存储experience

    def select_action(self, clusters, X, feature_names, op=None, cached_state_embed=None, cached_cluster_state=None,
                      for_next=False, steps_done=0):
        if self.select_mode:  # assert is head mode
            return self._select_head(clusters, X, feature_names, cached_state_embed, cached_cluster_state,
                                     for_next=for_next, steps_done=steps_done)
        else:
            assert op is not None
            return self._select_tail(clusters, X, feature_names, op, cached_state_embed, cached_cluster_state,
                                     for_next=for_next, steps_done=steps_done)

    def _select_head(self, clusters, X, feature_names, cached_state_embed=None, cached_cluster_state=None,
                     for_next=False, steps_done=0):
        #cluster应该是选出的一组features
        #x应该是原始feature set
        q_vals, select_cluster_state_list, state_op_emb = self.forward(clusters, X, for_next=for_next)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        if for_next:
            acts = np.argmax(q_vals)
            # m = Categorical(prob)
            # acts = m.sample()  # on the self._device
            # acts = acts.item()  # tensor shape [1] => a number (int)
        else:
            if np.random.uniform() > eps_threshold:
                acts = np.argmax(q_vals)
                # m = Categorical(prob)
                # acts = m.sample()  # on the self._device
                # acts = acts.item()  # tensor shape [1] => a number (int)
            else:
                acts = np.random.randint(0, len(clusters))
        f_cluster = X[:, list(clusters[acts])]
        action_emb = select_cluster_state_list[acts]
        f_names = np.array(feature_names)[list(clusters[acts])]
        info('current select feature name : ' + str(f_names))
        return acts, action_emb, f_names, f_cluster, select_cluster_state_list, state_op_emb

    def _select_tail(self, clusters, X, feature_names, op, cached_state_embed, cached_cluster_state,
                     for_next=False, steps_done=0):
        op_emb = self.get_op_emb(op)
        op_emb = torch.tensor(op_emb)
        if self.cuda_info:
            op_emb = op_emb.cuda()
        state_op_emb = torch.cat((cached_state_embed, op_emb))
        q_vals, select_cluster_state_list, state_op_emb = self.forward(cached_state_emb=state_op_emb,
                                                                     cached_cluster_state=cached_cluster_state, for_next=for_next)  # act_probs: [bs, act_dim], state_value: [bs, 1]
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        if for_next:
            acts = np.argmax(q_vals)
            # m = Categorical(prob)
            # acts = m.sample()  # on the self._device
            # acts = acts.item()  # tensor shape [1] => a number (int)
        else:
            if np.random.uniform() > eps_threshold:
                acts = np.argmax(q_vals)
                # m = Categorical(prob)
                # acts = m.sample()  # on the self._device
                # acts = acts.item()  # tensor shape [1] => a number (int)
            else:
                acts = np.random.randint(0, len(clusters))
        f_cluster = X[:, list(clusters[acts])]
        action_emb = select_cluster_state_list[acts]
        f_names = np.array(feature_names)[list(clusters[acts])]
        info('current select feature name : ' + str(f_names))
        return acts, action_emb, f_names, f_cluster, select_cluster_state_list, state_op_emb

    # 𝐿 =∑𝑙𝑜𝑔𝜋𝜃(𝑠𝑡, 𝑎𝑡)(𝑟 + 𝛾𝑉(𝑠𝑡 + 1)−𝑉(𝑠𝑡))
    def learn(self, optimizer,op_mem=None):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_, b_a_ = self.memory.sample()
        info(f'Sampled memory : s1: {b_s.shape} ; b_a: {b_a.shape} ; b_r: {b_r.shape} ; b_s_: {b_s_.shape} ; b_a_: {b_a_.shape}')
        if self.select_mode == 'head':
            net_input = torch.cat((b_s, b_a), axis=1)
        else:
            net_input = torch.cat((b_s, b_a), axis=1) #这里需要添加state_op emb,aka, memory 部分也得改？或者我如果不改，那么应该直接就能跑，把op-dim部分修一下
        q_eval = self.eval_net(net_input) # 8*64 128*128 矩阵维度不对应
        net_input_ = torch.cat((b_s_, b_a_), axis=1)
        q_next = self.target_net(net_input_)
        q_target = b_r + self.GAMMA * q_next.view(self.BATCH_SIZE, 1) #这里没有定义batch_size,
        loss = self.loss_func(q_eval, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # actor_loss = []
        # critic_loss = []
        # entropy_loss = []
        # for index, cached_cluster_state in enumerate(alist):
        #     prob, _, _ = self.forward(cached_state_emb=s1[index], cached_cluster_state=cached_cluster_state)
        #     m = Categorical(prob)
        #     logp = m.log_prob(a[index])
        #     state_value1 = self.critic.forward(s1[index])
        #     state_value2 = self.critic.forward(s2[index])
        #     advantage = r[index] + self.GAMMA * state_value2 - state_value1
        #     actor_loss.append(advantage.detach() * -logp)
        #     critic_loss.append(advantage.pow(2))
        #     entropy_loss.append(m.entropy().unsqueeze(0))
        # actor_loss = torch.mean(torch.cat(actor_loss, 0))
        # critic_loss = torch.mean(torch.cat(critic_loss, 0))
        # entropy_loss = torch.mean(torch.cat(entropy_loss, 0))

        # loss = actor_loss + critic_loss + self.ENT_WEIGHT * entropy_loss

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        # if self.select_mode:
        #     name = 'model_c1'
        # else:
        #     name = 'model_c2'
        # info(
        #     f' | name={name}' +
        #     ' | loss={:.5f}'.format(loss.cpu().item()) +
        #     ' | ploss={:.5f}'.format(actor_loss.cpu().item()) +
        #     ' | vloss={:.5f}'.format(critic_loss.cpu().item()) +
        #     ' | entropy={:.5f}'.format(entropy_loss.cpu().item()) +
        #     f' | reward={r.cpu()}')


class OpNet(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS, N_HIDDEN, init_w):
        super(OpNet, self).__init__()
        self.fc1 = nn.Linear(N_STATES, N_HIDDEN)
        self.fc1.weight.data.normal_(0, init_w)
        self.out = nn.Linear(N_HIDDEN, N_ACTIONS)
        self.out.weight.data.normal_(0, init_w)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value

class OpDQNNetwork(DQNNetwork):
    def __init__(self, state_dim, cluster_state_dim, hidden_dim, memory: Replay, ent_weight,
                 gamma=0.99, device=None, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, init_w=1e-6):
        super(OpDQNNetwork, self).__init__(state_dim, cluster_state_dim, hidden_dim, gamma, device,
                                            memory, ent_weight, EPS_START=EPS_START, EPS_END=EPS_END,
                                            EPS_DECAY=EPS_DECAY, init_w=init_w)

        self.eval_net, self.target_net = OpNet(self.state_dim, OP_DIM, self.hidden_dim, init_w), \
                                    OpNet(self.state_dim, OP_DIM, self.hidden_dim, init_w)
        

    def forward(self, cluster_state, for_next=False):
        if for_next:
            return self.target_net.forward(cluster_state)
        else :
            return self.eval_net.forward(cluster_state)

    # def get_value(self, cluster_state):
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     action_value = self.out(x)
    #     return action_value

    def select_operation(self, select_cluster_state, for_next=False, steps_done=0):
        q_vals = self.forward(select_cluster_state, for_next)
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END
                                        ) * math.exp(-1.0 * steps_done / self.EPS_DECAY)
        q_vals = q_vals.detach()
        if for_next:
            acts = np.argmax(q_vals)
        else:
            if np.random.uniform() > eps_threshold:
                acts = np.argmax(q_vals)
            else:
                acts = np.random.randint(0, OP_DIM)
        op_name = operation_set[acts]
        info('current select op : ' + str(op_name))
        return acts, op_name

    def store_transition(self, s1, op, r, s2):
        self.memory.store_transition((s1, op, r, s2))

    # 𝐿 =∑𝑙𝑜𝑔𝜋𝜃(𝑠𝑡, 𝑎𝑡)(𝑟 + 𝛾𝑉(𝑠𝑡 + 1)−𝑉(𝑠𝑡))
    def learn(self, optimizer):
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        b_s, b_a, b_r, b_s_ = self.memory.sample()
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # prob = self.forward(s1)
        # m = Categorical(prob)
        # logp = m.log_prob(op.reshape(-1)).reshape(-1, 1)
        # state_value1 = self.get_value(s1)
        # state_value2 = self.get_value(s2)
        # advantage = r + self.GAMMA * state_value2 - state_value1
        # actor_loss = advantage.detach() * -logp
        # critic_loss = advantage.pow(2)
        # entropy_loss = m.entropy()
        # actor_loss = actor_loss.mean()
        # critic_loss = critic_loss.mean()
        # entropy_loss = entropy_loss.mean()
        # loss = actor_loss + critic_loss + self.ENT_WEIGHT * entropy_loss
        # optimizer.zero_grad()
        # loss.backward()
        #optimizer.step()
        # info(
        #     ' | name=model_op' +
        #     ' | loss={:.5f}'.format(loss.cpu().item()) +
        #     ' | ploss={:.5f}'.format(actor_loss.cpu().item()) +
        #     ' | vloss={:.5f}'.format(critic_loss.cpu().item()) +
        #     ' | entropy={:.5f}'.format(entropy_loss.cpu().item()) +
        #     f' | reward={r.cpu()}')

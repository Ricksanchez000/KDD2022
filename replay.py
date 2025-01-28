from collections import defaultdict

import numpy as np
import torch


class Replay:
    def __init__(self, size, batch_size, device):
        self.MEMORY_CAPACITY = size
        self.memory_counter = 0
        self.BATCH_SIZE = batch_size
        self.cuda_info = device is not None

    def _sample(self):
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        return sample_index

    def sample(self):
        raise NotImplementedError()

    def store_transition(self, resource):
        raise NotImplementedError()


class RandomClusterReplay(Replay):
    def __init__(self, size, batch_size, state_shape, device, op_dim=0):
        super().__init__(size, batch_size, device)
        self.memory = np.zeros((self.MEMORY_CAPACITY, state_shape * 2 + state_shape *
                                2 + op_dim * 2 + 1))
        
        #所以，要么后续传入fc1计算q_value有问题，要么改这里的DIM
        #因为没有理由传入128的维度？64和64+19 = state_dim + op_dim ? 那问题在于，第一个agent接收的输入是什么？
        #哦！我明白了，第一个agent接收的是REP(feature_set)=64
        #op_agent接收的是REP(F1) cat REP(Ct) = 64
        #tail_agent接收的是REP(F1) cat REP(o1) cat REP (C1)
        #所以，第一个agent接收的 REP(F1)应该是 64
        
        #self.memory = np.zeros((self.MEMORY_CAPACITY, state_shape * 2 + op_dim * 2 + 1)) #64*2 + 15*2 + 1 = 159 
        #如果op_dim=0，则64*2 + 0*2 + 1 = 129
        #不行，经过检查发现存储了a1,维度是64，所以需要是64*2 + 64*2 = 1 = 257
        #如果是op_dim = 15,则是64*2 + 64*2 + 15*2 = 287 是在
        
        #所以现在需要看的是a为什么需要占64的维度？
        
        
        
        self.STATE_DIM = state_shape
        self.ACTION_DIM = state_shape
        self.op_dim = op_dim
        if self.cuda_info:
            self.mem1 = self.mem1.cuda()
            self.mem2 = self.mem2.cuda()
            self.reward = self.reward.cuda()

    def store_transition(self, mems):
        s, a, r, s_, a_ = mems

        transition = np.hstack((s, a, [r], s_, a_)) #这里store好像是没什么问题，都是dim = 64
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        

    def sample(self):
        sample_index = self._sample()
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.STATE_DIM + self.op_dim])
        b_a = torch.LongTensor(b_memory[:, self.STATE_DIM + self.op_dim :self.STATE_DIM + self.op_dim
                                                        +  self.ACTION_DIM])
        b_r = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.op_dim + self.
                                ACTION_DIM:self.STATE_DIM + self.op_dim + self.ACTION_DIM + 1]) # only 1 slice，contails the simgle reward
        b_s_ = torch.FloatTensor(b_memory[:, self.STATE_DIM + self.op_dim + self.ACTION_DIM + 1:self.STATE_DIM * 2 + self.op_dim*2 self.ACTION_DIM + 1])
        b_a_ = torch.LongTensor(b_memory[:, -self.ACTION_DIM:])
        return b_s, b_a, b_r, b_s_, b_a_

        '''
        b_s = b_memory[:, :64]         # (batch_size, 64)
        b_a = b_memory[:, 64:128]      # (batch_size, 64)
        b_r = b_memory[:, 128:129]     # (batch_size, 1)
        b_s_ = b_memory[:, 129:193]    # (batch_size, 64)
        b_a_ = b_memory[:, 193:257]    # (batch_size, 64)
        '''
        
class RandomOperationReplay(Replay):
    def __init__(self, size, batch_size, state_dim, device):
        super().__init__(size, batch_size, device)
        self.memory = np.zeros((self.MEMORY_CAPACITY, state_dim * 2 + 2))
        self.N_STATES = state_dim

    def store_transition(self, mems):
        s1, op, r, s2 = mems
        transition = np.hstack((s1, [op, r], s2))
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
        # self.mem1[index] = s1
        # self.mem2[index] = s2
        # self.reward[index] = r
        # self.op[index] = op
        # self.memory_counter += 1

    def sample(self):
        sample_index = self._sample()
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES])
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES + 1:self.N_STATES +
                                                              2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:])
        return b_s, b_a, b_r, b_s_


class PERClusterReplay(RandomClusterReplay):
    def __init__(self, size, state_dim, action_dim, batch_size):
        super().__init__(size, state_dim, action_dim, batch_size)

    def _sample(self):
        raise NotImplementedError()

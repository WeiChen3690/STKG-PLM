# -*- coding: utf-8 -*-
import numpy as np
import datetime
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import argparse
import os
import pickle
from torch.autograd import Variable
import collections
import pandas as pd
from torch_geometric.data import Data
import random
from tqdm import tqdm
from torch.utils.data import Dataset
import transformers
from collections import deque


class KG_data(Dataset):
    def __init__(self, args):
        self.p = args
        self.args = args
        self.entity_num_per_item = 100
        self.load_data()
        self.kg_dict, self.heads = self.generate_kg_data()
        self.device = torch.device("cuda:{}".format(self.args.gpu_id))
        
    def load_data(self):
        # triple
        self.src = []
        self.rel = []
        self.dst = []
        self.user_list = []
        self.loc_list = []
        # read files, note: Gowalla don't have it
        if self.args.data_dir != './datasets/Foursquare_GWL/':
            with open(self.args.data_dir + 'triple_pc.txt', 'r') as f:      # poi-belong to-category
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n').split('\t')
                    poi, belong, category = line
                    self.src.append(int(poi))
                    self.rel.append(int(belong))
                    self.dst.append(int(category))
                    self.loc_list.append(int(poi))
        with open(self.args.data_dir + 'triple_ptp.txt', 'r') as f:     # poi-next to-poi
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                poi1, to, poi2 = line
                self.src.append(int(poi1))
                self.rel.append(int(to))
                self.dst.append(int(poi2))
        with open(self.args.data_dir + 'triple_utp.txt', 'r') as f:     # user-visit-poi
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                uid, visit, poi = line
                self.src.append(int(uid))
                self.rel.append(int(visit))
                self.dst.append(int(poi))
                self.user_list.append(int(uid))
        self.num_nodes = len(set(self.src + self.dst))     # num of nodes
        self.num_rels = len(set(self.rel))                 # num of relations
        self.num_users = len(set(self.user_list))          # num of users
        self.num_locs = len(set(self.dst))                      # num of locations
        self.kg_data = pd.read_csv(self.args.data_dir + 'kg.txt', sep=' ', names=['h', 'r', 't'], engine='python')
    
    def construct_adj(self):
        # construct adj matrix
        edge_index, edge_type = [], []
        for s, r, d in zip(self.src, self.rel, self.dst):  # no inverse edge
            edge_index.append((s, d))
            edge_type.append(r)
        edge_index = torch.LongTensor(edge_index).t()	                  # (2, num_edge) graph
        edge_type = torch.LongTensor(edge_type).reshape(len(self.rel), 1)  # (num_edge, 1)
        return edge_index, edge_type
    
    def generate_kg_data(self):
        kg_dict = collections.defaultdict(list)
        for s,r,d in zip(self.src, self.rel, self.dst):
            kg_dict[s].append((r,d))                        # src -> [(rel1, dst1), (rel2, dst2)]
        heads = list(kg_dict.keys())
        return kg_dict, heads

    # 采样为了数据对齐
    def sample_align(self, head_num):           # kg_dict: {head:(rel, tail)}
        neighbor_num = self.entity_num_per_item             # 节点的个数
        i2es = dict()                                       # {head: tail0, tail1}->kg_dict
        i2rs = dict()                                       # {head: rel0, rel1}
        for item in range(head_num + 1):
            rts = self.kg_dict.get(item, False)             
            if rts:
                tails = list(map(lambda x:x[1], rts))       # 当前node的所有tails
                relations = list(map(lambda x:x[0], rts))   # 当前node的所有relation

                if(len(tails) > neighbor_num):
                    i2es[item] = tails[:neighbor_num]
                    i2rs[item] = relations[:neighbor_num]
                else:
                    # 0 as padding idx
                    tails.extend([0]*(neighbor_num-len(tails)))
                    relations.extend([0]*(neighbor_num-len(relations)))
                    i2es[item] = tails
                    i2rs[item] = relations
            else:
                i2es[item] = [0]*neighbor_num
                i2rs[item] = [0]*neighbor_num
        i2es_t = dict()
        for item in range(head_num + 1):
            rts = self.kg_dict.get(item, False)             
            if rts:
                tails = list(map(lambda x:x[1], rts))       # 当前node的所有tails
                if(len(tails) > neighbor_num):
                    i2es_t[item] = torch.IntTensor(tails)[:neighbor_num]
                else:
                    # 0 as padding idx
                    tails.extend([0]*(neighbor_num-len(tails)))
                    i2es_t[item] = torch.IntTensor(tails)
            else:
                i2es_t[item] = torch.IntTensor([0]*neighbor_num)
        return i2es, i2es_t, i2rs
    
    def __len__(self):
        return len(self.kg_dict)
    
    def __getitem__(self, index):
        head = self.heads[index]
        relation, pos_tail = random.choice(self.kg_dict[head])
        while True:
            neg_head = random.choice(self.heads)
            neg_tail = random.choice(self.kg_dict[neg_head])[1]
            if (relation, neg_tail) in self.kg_dict[head]:
                continue
            else:
                break
        return head, relation, pos_tail, neg_tail


class Sequence_data(Dataset):
    def __init__(self, args):
        self.args = args
    
    def generate_queue(self, train_idx, mode, mode2):
        """
        return a deque. You must use it by train_queue.popleft()
        content: [(uid, session_id in uid)...]
        train_idx:  dict, uid0:[pid0, pid1, pid2....]
        mode:   random
        mode2:  train or test
         """
        user = list(train_idx.keys())                               # 所有的uid(list)
        train_list = []
        if mode == 'random':                                        # 打乱放入序列，放入都是tuple(uid, session id)
            initial_queue = {}
            for u in user:
                if mode2 == 'train':                                # train set
                    initial_queue[u] = deque(train_idx[u][1:])      # {uid0:deque[session1,session2,...], uid1:....}
                else:                                               # test set 
                    initial_queue[u] = deque(train_idx[u])          # {uid0:deque[session0]}
            queue_left = 1
            while queue_left > 0:                                   # 每次都取打乱后user的前1/100->train_queue
                np.random.shuffle(user)                              # uid打乱
                for j, u in enumerate(user):
                    if len(initial_queue[u]) > 0:
                        seq = {}
                        seq[u] = initial_queue[u].popleft()
                        train_list.append(seq)
                    if j >= int(0.01 * len(user)):
                        break
                queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])   # 直到都pop掉了再跳出，每次pop掉某一个uid的最left的session id
        elif mode == 'normal':                                      # 依次放入队列 （uid, session id)
            for u in user:
                for i in train_idx[u]:
                    seq = {}
                    seq[u] = i
                    train_list.append(seq)
        return train_list                                          # deque([], [])

    def generate_input_history(self, data_neural, mode, candidate=None):
        row = {}  # row:{uid0:{0:{'loc':,'time':,'word':}...}...}  
        u_list = []
        train_idx = {}
        if candidate is None:                                                                       # candidate: uid list (NYC:1020)
            candidate = data_neural.keys()
        for u in list(candidate):
            # sessions = data_neural[u]['sessions_with_word']                                       # (pid, tid, wid) total sessions of u, 含编码后的时间
            sessions = data_neural[u]['sessions_with_word']                                         # 还有编码后的 tid
            session_id = data_neural[u]['train'] + data_neural[u]['test']                           # sessions of u(list of session id)，time: np.float
            train_id = data_neural[u][mode]
            row[u] = []
            for c, i in enumerate(session_id):                                                      # 从第二个session开始遍历
                session = sessions[i]                                                               # session:  (pid, tid, wid)
                trace = {}
                # 这里不区分target，先整合uid, 序列\n数据格式
                loc_list = [s[0] for s in session]                                                  # session中的pid
                tim_list = [s[1] for s in session]                                                  # session中的time id
                word_list = [s[2] for s in session]                                                 # session中的word id
                if len(loc_list) < 5:
                    continue
                trace['loc'] = loc_list
                trace['time'] = tim_list
                trace['word_list'] = word_list
                row[u].append(trace)
                u_list.append(u)
            train_idx[u] = train_id
        return row, u_list, train_idx     

    # 这里不要忘了data_pre也需要更该，验证集测试集都要有
    def POI_data_pre(self):                                                                             # 生成类似home.txt的文件以及train.pkl, valid.pkl, test.pkl
        data = pickle.load(open(self.args.data_dir + 'foursquare_' + self.args.dataset_name + '_4input.pkl', 'rb'))
        data_neural = data['data_neural']
        trans2id_dict = data['trans2id']
        candidate = data_neural.keys()
        row, u_list, train_idx = self.generate_input_history(data_neural, 'train', candidate)
        row, u_list, test_idx = self.generate_input_history(data_neural, 'test', candidate)
        row, u_list, valid_idx = self.generate_input_history(data_neural, 'valid', candidate)
        train_id = self.generate_queue(train_idx, 'random', 'train')          # {uid: session_id, uid1: session_id}
        test_id = self.generate_queue(test_idx, 'random', 'test')
        valid_id = self.generate_queue(valid_idx, 'random', 'valid')
        loc_row_overall = []
        loc_row_train = []                                               # loc_row是dict的list，每一个dict是一row
        loc_row_train_trans = []
        loc_row_train_category = []
        loc_row_test = []
        loc_row_test_trans = []
        loc_row_test_category = []
        loc_row_valid = []
        loc_row_valid_trans = []
        loc_row_valid_category = []

        # 生成new tokens for bert
        new_tokens = []
        entity_id_lst = []
        with open(self.args.data_dir + 'entity_user_dict.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                entity_id = line[0]
                st = '[User{}]'.format(entity_id)
                new_tokens.append(st)
                entity_id_lst.append(int(entity_id))
        with open(self.args.data_dir + 'entity_loc_dict.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip('\n').split('\t')
                entity_id = line[0]
                st = '[POI{}]'.format(entity_id)
                new_tokens.append(st)
                entity_id_lst.append(int(entity_id))

        # 对train_id、test_id、valid_id进行处理，将session_id替换为真实的session序列 all sequence
        for dic in train_id:
            uid = int(list(dic.keys())[0])
            session_id = int(list(dic.values())[0])
            sessions_lookup = data_neural[uid]['sessions_with_word']
            sessions_trans_lookup = data_neural[uid]['sessions_trans']
            session = [x[0] for x in sessions_lookup[session_id]]
            session_cate = [x[2] for x in sessions_lookup[session_id]]
            session_trans = [x[1] for x in sessions_trans_lookup[session_id]]
            seq = {}
            seq_trans = {}
            seq_cate = {}
            seq[uid] = session
            seq_trans[uid] = session_trans
            seq_cate[uid] = session_cate
            loc_row_train_trans.append(seq_trans)
            loc_row_train.append(seq)
            loc_row_train_category.append(seq_cate)
            loc_row_overall.append(seq)
        for dic in test_id:
            uid = int(list(dic.keys())[0])
            session_id = int(list(dic.values())[0])
            sessions_lookup = data_neural[uid]['sessions_with_word']
            sessions_trans_lookup = data_neural[uid]['sessions_trans']
            session = [x[0] for x in sessions_lookup[session_id]]
            session_cate = [x[2] for x in sessions_lookup[session_id]]
            session_trans = [x[1] for x in sessions_trans_lookup[session_id]]
            seq = {}
            seq_trans = {}
            seq_cate = {}
            seq[uid] = session
            seq_trans[uid] = session_trans
            seq_cate[uid] = session_cate
            loc_row_test_trans.append(seq_trans)
            loc_row_test.append(seq)
            loc_row_test_category.append(seq_cate)
            loc_row_overall.append(seq)
        for dic in valid_id:
            uid = int(list(dic.keys())[0])
            session_id = int(list(dic.values())[0])
            sessions_lookup = data_neural[uid]['sessions_with_word']
            sessions_trans_lookup = data_neural[uid]['sessions_trans']
            session = [x[0] for x in sessions_lookup[session_id]]
            session_cate = [x[2] for x in sessions_lookup[session_id]]
            session_trans = [x[1] for x in sessions_trans_lookup[session_id]]
            seq = {}
            seq_trans = {}
            seq_cate = {}
            seq[uid] = session
            seq_trans[uid] = session_trans
            seq_cate[uid] = session_cate
            loc_row_valid_trans.append(seq_trans)
            loc_row_valid.append(seq)
            loc_row_valid_category.append(seq_cate)
            loc_row_overall.append(seq)
        # formulate nyc.txt
        with open(self.args.data_dir + 'overall.txt', 'w') as f:
            for dic in loc_row_overall:
                uid = int(list(dic.keys())[0])
                seq = dic[uid]
                seq_str = str(seq).strip('[]').replace(', ', ',')
                f.write(str(uid) + ' ' + seq_str + '\n')
        # formulate nyc_train.txt
        # with open(overall_path + 'nyc_train.txt', 'w') as f:
        #     for dic in loc_row_train:   
        #         uid = int(list(dic.keys())[0])
        #         seq = dic[uid]
        #         seq_str = str(seq).strip('[]').replace(', ', ',')
        #         f.write(str(uid) + ' ' + seq_str + '\n')
        # formulate nyc_test.txt
        # with open(overall_path + 'nyc_test.txt', 'w') as f:
        #     for dic in loc_row_test:
        #         uid = int(list(dic.keys())[0])
        #         seq = dic[uid]
        #         seq_str = str(seq).strip('[]').replace(', ', ',')
        #         f.write(str(uid) + ' ' + seq_str + '\n')
        # formulate nyc_valid.txt
        # with open(overall_path + 'nyc_valid.txt', 'w') as f:
        #     for dic in loc_row_valid:
        #         uid = int(list(dic.keys())[0])
        #         seq = dic[uid]
        #         seq_str = str(seq).strip('[]').replace(', ', ',')
        #         f.write(str(uid) + ' ' + seq_str + '\n')
        # formulate all_train_seq.txt, for building the graph
        # with open(overall_path + 'all_train_seq.txt', 'w') as f:
        #     for dic in loc_row_train:
        #         uid = int(list(dic.keys())[0])
        #         seq = dic[uid]
        #         seq_str = str(seq).strip('[]').replace(', ', ',')
        #         f.write(str(uid) + ' ' + seq_str + '\n')
        #     for dic in loc_row_valid:
        #         uid = int(list(dic.keys())[0])
        #         seq = dic[uid][:-1]
        #         seq_str = str(seq).strip('[]').replace(', ', ',')
        #         f.write(str(uid) + ' ' + seq_str + '\n')
        #     for dic in loc_row_test:
        #         uid = int(list(dic.keys())[0])
        #         seq = dic[uid][:-1]
        #         seq_str = str(seq).strip('[]').replace(', ', ',')
        #         f.write(str(uid) + ' ' + seq_str + '\n')
        # formulate all_train_seq_user.txt, for building the graph
        # loc_uid_dict = {}
        # all_train_seq_user = {}         # {pid: [uid0, uid1, uid2..]} 访问过pid的uid
        # for dic in loc_row_overall:
        #     uid = int(list(dic.keys())[0])
        #     seq = dic[uid]
        #     for poi in seq:
        #         if poi not in loc_uid_dict:
        #             loc_uid_dict[poi] = []
        #             loc_uid_dict[poi].append(uid)
        #         elif poi in loc_uid_dict and uid not in loc_uid_dict[poi]:
        #             loc_uid_dict[poi].append(uid)
        # for pid in loc_uid_dict:                                # filter
        #     if len(loc_uid_dict[pid]) >= 2:
        #         all_train_seq_user[pid] = loc_uid_dict[pid]
        # with open(overall_path + 'all_train_seq_user.txt', 'w') as f:
        #     for poi in all_train_seq_user.keys():
        #         poi = int(poi)
        #         user_seq = all_train_seq_user[poi]
        #         user_seq_str = str(user_seq).strip('[]').replace(', ', ',')
        #         f.write(str(poi) + ' ' + user_seq_str + '\n')
        # formulate train.pkl
        train_pkl_u_list = []
        train_pkl_seq = []
        train_pkl_seq_len = []
        train_pkl_target = []
        train_pkl_trans = []
        train_pkl_cate = []
        for dic, dic_trans, dic_cate in zip(loc_row_train, loc_row_train_trans, loc_row_train_category):
            uid = int(list(dic.keys())[0])
            seq = dic[uid][:-1]
            seq_trans = dic_trans[uid][:-1]
            seq_cate = dic_cate[uid][:-1]
            seq_trans = [trans2id_dict[x] + 50 for x in seq_trans]
            seq_trans.insert(0, 0)
            seq_length = len(seq)
            train_pkl_seq.append(seq)
            train_pkl_target.append(dic[uid][-1])
            train_pkl_seq_len.append(seq_length)
            train_pkl_u_list.append(uid)
            train_pkl_trans.append(seq_trans)
            train_pkl_cate.append(seq_cate)
        train = (train_pkl_u_list, train_pkl_seq, train_pkl_target, train_pkl_seq_len, train_pkl_trans, train_pkl_cate)
        pickle.dump(train, open(self.args.data_dir + 'train.pkl', 'wb'))
        # formulate valid.pkl
        valid_pkl_u_list = []
        valid_pkl_seq = []
        valid_pkl_seq_len = []
        valid_pkl_target = []
        valid_pkl_trans = []
        valid_pkl_cate = []
        for dic, dic_trans, dic_cate in zip(loc_row_valid, loc_row_valid_trans, loc_row_valid_category):
            uid = int(list(dic.keys())[0])
            seq = dic[uid][:-1]
            seq_trans = dic_trans[uid][:-1]
            seq_cate = dic_cate[uid][:-1]
            seq_trans = [trans2id_dict[x] + 50 for x in seq_trans]
            seq_trans.insert(0, 0)
            valid_pkl_u_list.append(uid)
            valid_pkl_seq.append(seq)
            valid_pkl_seq_len.append(len(seq))
            valid_pkl_target.append(dic[uid][-1])
            valid_pkl_trans.append(seq_trans)
            valid_pkl_cate.append(seq_cate)
        valid = (valid_pkl_u_list, valid_pkl_seq, valid_pkl_target, valid_pkl_seq_len, valid_pkl_trans, valid_pkl_cate)
        pickle.dump(valid, open(self.args.data_dir + 'valid.pkl', 'wb'))
        # formulate test.pkl
        test_pkl_u_list = []
        test_pkl_seq = []
        test_pkl_seq_len = []
        test_pkl_target = []
        test_pkl_trans = []
        test_pkl_cate = []
        for dic, dic_trans, dic_cate in zip(loc_row_test, loc_row_test_trans, loc_row_test_category):
            uid = int(list(dic.keys())[0])
            seq = dic[uid][:-1]
            seq_trans = dic_trans[uid][:-1]  
            seq_cate = dic_cate[uid][:-1]
            seq_trans = [trans2id_dict[x] + 50 for x in seq_trans]
            seq_trans.insert(0, 0)
            test_pkl_u_list.append(uid)
            test_pkl_seq.append(seq)
            test_pkl_seq_len.append(len(seq))
            test_pkl_target.append(dic[uid][-1])
            test_pkl_trans.append(seq_trans)
            test_pkl_cate.append(seq_cate)
        test = (test_pkl_u_list, test_pkl_seq, test_pkl_target, test_pkl_seq_len, test_pkl_trans, test_pkl_cate)
        pickle.dump(test, open(self.args.data_dir + 'test.pkl', 'wb'))
        return train_id, test_id, valid_id, new_tokens, entity_id_lst
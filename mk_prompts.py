from torch import Tensor
from dataset import Dataset as DatasetFromMe
import transformers
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForMaskedLM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from torch.nn.parallel import DistributedDataParallel as DDP
# from bertviz import model_view, head_view
from utils import *
import warnings
import torch.nn.functional as F
import pickle
from num2words import num2words

# 首先要获得bert data
class MakePrompts:
    def __init__(self, train_id, test_id, valid_id, datasets_name):
        self.dataTrain = []
        self.dataTrain_ptp = []
        self.dataTrain_time = []
        self.dataValid = []
        self.dataValid_ptp = []
        self.dataValid_time = []
        self.dataTest = []
        self.dataTest_ptp = []
        self.dataTest_time = []
        self.datasets_name = datasets_name
        self.overall_path = './datasets/Foursquare_' + self.datasets_name + '/'
        # get bert data
        data = pickle.load(open(self.overall_path + 'foursquare_' + self.datasets_name + '_4input.pkl', 'rb'))
        self.data_temp = data['data_temp']
        self.data_neural = data['data_neural']
        self.time_max = data['time_max']
        self.time_min = data['time_min']
        self.dis_max = data['dis_max']
        # dataTrain
        for dic in train_id:
            uid = int(list(dic.keys())[0])
            session_id = int(list(dic.values())[0])
            sessions_utp_lookup = self.data_temp[uid]['sessions_utp_prompt']
            sessions_ptp_lookup = self.data_neural[uid]['sessions_trans']
            sessions_real_time = self.data_temp[uid]['sessions_real_time']
            session = [x for x in sessions_utp_lookup[session_id]]
            session_ptp = [x for x in sessions_ptp_lookup[session_id]]
            sessions_real_time = [x for x in sessions_real_time[session_id]]
            self.dataTrain.append(session)
            self.dataTrain_ptp.append(session_ptp)
            self.dataTrain_time.append(sessions_real_time)
        for dic in test_id:
            uid = int(list(dic.keys())[0])
            session_id = int(list(dic.values())[0])
            sessions_utp_lookup = self.data_temp[uid]['sessions_utp_prompt']
            sessions_ptp_lookup = self.data_neural[uid]['sessions_trans']
            sessions_real_time = self.data_temp[uid]['sessions_real_time']
            session = [x for x in sessions_utp_lookup[session_id]]
            session_ptp = [x for x in sessions_ptp_lookup[session_id]]
            sessions_real_time = [x for x in sessions_real_time[session_id]]
            self.dataTest.append(session)
            self.dataTest_ptp.append(session_ptp)
            self.dataTest_time.append(sessions_real_time)
        for dic in valid_id:
            uid = int(list(dic.keys())[0])
            session_id = int(list(dic.values())[0])
            sessions_utp_lookup = self.data_temp[uid]['sessions_utp_prompt']
            sessions_ptp_lookup = self.data_neural[uid]['sessions_trans']
            sessions_real_time = self.data_temp[uid]['sessions_real_time']
            session = [x for x in sessions_utp_lookup[session_id]]
            session_ptp = [x for x in sessions_ptp_lookup[session_id]]
            sessions_real_time = [x for x in sessions_real_time[session_id]]
            self.dataValid.append(session)
            self.dataValid_ptp.append(session_ptp)
            self.dataValid_time.append(sessions_real_time)
        # self.dataTrain = torch.tensor(self.dataTrain)                                               # [ [[utp0], [utp1]...], [...], ... ]
        # self.dataValid = torch.tensor(self.dataValid)
        # self.dataTest = torch.tensor(self.dataTest)

        self.id_rel = {}
        for i in range(0, 145):
            self.id_rel[i] = 'visit'        # check in
        self.id_rel[-1] = 'visit'
        self.time_dict = {}
        self.get_time_dict()

    def get_time_dict(self):
        with open(self.overall_path + 'time_dict.txt', 'r') as f:
            rows = f.readlines()
            for row in rows:
                seq = row.replace('\n', '').split(':')
                self.time_dict[int(seq[0])] = seq[1]
    
    def triple2sentence_single(self, triple, i, x, dis, real_time): 
        mode = 'normal'
        uid = int(triple[0])
        rid = int(triple[1])                    # time interval
        dis = int(dis)
        time_stamp = int(triple[1])
        pid = int(triple[2]) 

        head = '[User{}]'.format(uid)
        if i == 0:
            dis_str = ''
        else:
            # 将距离转换为sentence
            if dis == 201:
                word_dis = 'and a longer distance'
            else:
                word_dis = num2words(int(dis * 100))                              
            dis_str = 'and a distance of {} meters'.format(word_dis)     

        rel = self.id_rel[rid]                                                      # relation sentence 2 idx
        tail = '[POI{}]'.format(pid)
        tim = '[STR] ' + self.time_dict[time_stamp]                                 # 转换成时间sentence, 前面加上ST-Relation
        
        if i == 0:
            return '[CHE] At the beginning' + ' ' + head + ' ' + rel + ' ' + tail + ' at ' + real_time + ' ; '
        elif i == len(x) - 1:
            return '[CHE]' + ' ' + 'At the end' + ' ' + head + ' ' + rel + ' ' + tail + ' ; '
        else:
            if mode == 'test_time':
                return dis_str.replace('and a', 'after a') + ' ' + head + ' ' + rel + ' ' + tail + ' ; '
            else:
                # return tim  + ' ' + dis_str + ' ' + head + ' ' + rel + ' ' + tail + ' at ' + real_time + ' ; '
                return tim  + ' ' + dis_str + ' ' + head + ' ' + rel + ' ' + tail + ' ; '

    def triple2sentence(self, x, ptp_session, time_session):
        sentence = []
        path = []
        i = 0
        real_time = [i.split(' ')[1] for i in time_session]
        for triple, triple_real_time in zip(x, real_time):
            if i == 0:
                dis = -1
            else:
                dis = ptp_session[i - 1][1][1]
            sentence.append(self.triple2sentence_single(triple, i, x, dis, triple_real_time))
            path.append(triple)
            i += 1
            text = ''.join(sentence)
            text = ' [TRA] ' + text
        return text, path
    
    def sample(self, triple_session, ptp_session, time_session):
        triple_session = torch.IntTensor(triple_session)
        triple_session[1:, 1] = triple_session[1:,1] - triple_session[:-1,1]                                # delta t
        triple_session[0, 1] = -1
        text, path = self.triple2sentence(triple_session, ptp_session, time_session)                        # 转化为sentence
        return text, int(path[-1][2]), path
    
    def triple_sample(self, mode):
        if mode == 'train':
            data = self.dataTrain                                       # utp
            data_ptp = self.dataTrain_ptp                               # ptp
            data_time = self.dataTrain_time
        elif mode == 'valid':
            data = self.dataValid
            data_ptp = self.dataValid_ptp
            data_time = self.dataValid_time
        else:
            data = self.dataTest
            data_ptp = self.dataTest_ptp
            data_time = self.dataTest_time
        train_text = []                                                 # 用于train的sentence
        ground_truth = []
        paths = []
        print('Making {} data...'.format(mode))
        for triple_session, ptp_session, time_session in tqdm(zip(data, data_ptp, data_time)):
            text, label, path = self.sample(triple_session, ptp_session, time_session) # label是采样得到的最后一个的三元组object，作为ground truth，也就是当前序列的下一个POI
            train_text.append(text)                                      # [ sentence_seq0, sentence_seq1, ..., sentence_seq N ]
            ground_truth.append(label)
            paths.append(path)
        ground_truth = torch.LongTensor(ground_truth)                   # path是对应的实体和关系id
        return train_text, ground_truth, paths                          # train_text是个list,每个元素为固定某个subject采样所得到所有图序列的sentence

    def get_data(self, mode):
        data, label, paths = self.triple_sample(mode)
        bert_data = {'data': data, 'label': label, 'paths': paths}      # 软提示？
        return bert_data
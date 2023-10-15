# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        x1 = self.dataset1[index]
        x2 = self.dataset2[index]
        return x1, x2

    def __len__(self):
        return len(self.dataset1)
    

class GCL4SRData(Dataset):
    def __init__(self, args, data, test_neg_items=None):
        self.args = args
        self.data = data
        self.test_neg_items = test_neg_items
        self.max_len = args.max_seq_length
        self.uid_list = data[0]
        self.part_sequence = data[1]                            # sequence[[],[]]
        self.part_sequence_target = data[2]                     # the target of every seq
        self.part_sequence_length = data[3]                     # the length of every seq
        self.part_trans_sequence = data[4]
        self.part_cate_sequence = data[5]
        # self.part_tim_sequence_target = data[5]
        self.length = len(data[0])

    def __getitem__(self, index):

        input_ids = self.part_sequence[index]                   # index:seq_id
        target_pos = self.part_sequence_target[index]   
        seq_length = self.part_sequence_length[index]     
        user_id = self.uid_list[index]
        trans_seq = self.part_trans_sequence[index]
        # cate_seq = self.part_cate_sequence[index]
        # tim_tar = self.part_tim_sequence_target[index]

        pad_len = self.max_len - len(input_ids)                 # 需要pad的长度 10 - len
        input_ids = input_ids + [0] * pad_len                   # padding
        
        trans_pad_len = self.max_len - len(trans_seq)
        trans_seq = trans_seq + [0] * trans_pad_len

        # cate_pad_len = self.max_len - len(cate_seq)
        # cate_seq = cate_seq + [0] * cate_pad_len

        # input_ids = input_ids[:seq_length]

        assert len(input_ids) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:                                                   # 转化为tensor
            cur_tensors = ( 
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(seq_length, dtype=torch.long),
                torch.tensor(trans_seq, dtype=torch.long)
                # torch.tensor(cate_seq, dtype=torch.long),
            )

        return cur_tensors

    def __len__(self):
        return self.length


class PreBertDataset(Dataset):
    def __init__(self, tokenizer, max_length, bert_data):
        super(PreBertDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = bert_data['data']                           # [sentence of entity1, sentence of entity2, ...]
        self.label = bert_data['label']                        # the last object of graph sequence with entity-i, ground truth
        self.paths = bert_data['paths']                         # 相应的四元组
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]
        path = self.paths[index]
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = inputs['input_ids'].flatten()

        tail_mask = input_ids.detach().clone()

        labels = input_ids.detach().clone()
        rand = torch.rand(input_ids.shape)
        # mask_arr = (rand < 0.7) * (input_ids != 101) * (input_ids != 102) * (input_ids != 103) * (input_ids >= 29613)        # 随机对User和POI分词进行mask，特殊标记不mask
        mask_arr = (rand < 0.4) * (input_ids != 104)* (input_ids != 105) * (input_ids != 105) * (input_ids != 1) * (input_ids != 2) * (input_ids != 3) * (input_ids != 4) * (input_ids != 5) * (input_ids != 6)
        # mask_arr = (rand < 0.4) * (input_ids != 103)* (input_ids != 104) * (input_ids != 105)
        selection = torch.flatten(mask_arr.nonzero()).tolist()                                  

        last_non_zero = torch.nonzero(labels, as_tuple=False)[-1]                                 # 找到sentence的最后一个非0word
        tail_pos = last_non_zero - 2
        tail_index = tail_mask[tail_pos]                                                          # tail token id
        input_ids[selection] = 107                                                           # 103是[MASK]  
        input_ids[tail_pos] = 107
        tail_mask[tail_pos] = 107

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': inputs['attention_mask'].flatten(),
            'tail_mask': tail_mask,
            'tail_index': tail_index,
            'tail_pos': tail_pos,
            'text': self.data[index]
        }

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
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForMaskedLM
import time

from dataset import GCL4SRData, MyDataset
from trainer import GCL4SR_Train
from model import Simple_GRU, STKG_PLM, Bert4PR, TransR_train
from utils import check_path, set_seed, EarlyStopping, get_matrix_and_num
from collections import deque
from mk_data import KG_data, Sequence_data
from mk_prompts import MakePrompts



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--save_name", default='STKG_PLM-TKY-9-20', type=str)                        # ckpt name---need change
    parser.add_argument("--dataset_name", default='TKY', type=str)                              # need change
    parser.add_argument("--train_model", type=str, default='GRU', help="GRU, STKG_PLM, Bert4PR")                              # change model here
    parser.add_argument("--load_ckpt", type=bool, default=False)                                # 是否load ckpt模型
    parser.add_argument("--do_eval", type=bool, default=False)
    parser.add_argument("--data_name", default='overall', type=str)
    parser.add_argument("--data_dir", default='./datasets/Foursquare_', type=str)
    parser.add_argument("--output_dir", default='output/', type=str)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    parser.add_argument("--TransR", type=bool, default=False)

    # optimizer
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate of adam")        # 0.005
    parser.add_argument("--lr_dc", type=float, default=0.7, help='learning rate decay.')
    parser.add_argument("--lr_dc_step", type=int, default=5,
                        help='the number of steps after which the learning rate decay.')
    parser.add_argument("--weight_decay", type=float, default=5e-5, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # transformer
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--hidden_act", default="gelu", type=str, help="activation function")
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.3, help="attention dropout")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout")

    # train args
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")    # 512
    parser.add_argument("--max_seq_length", default=10, type=int, help="max sequence length")  # 最长的序列长度为10 
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of model")
    parser.add_argument("--seed", default=2022, type=int, help="seed")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--patience", default=10, type=int, help="early stop patience")

    # graph neural network
    parser.add_argument("--gnn_dropout_prob", type=float, default=0.5, help="gnn dropout")
    parser.add_argument("--use_renorm", type=bool, default=True, help="use re-normalize when build witg")
    parser.add_argument("--use_scale", type=bool, default=False, help="use scale when build witg")
    parser.add_argument("--fast_run", type=bool, default=True, help="can reduce training time and memory")
    parser.add_argument("--sample_size", default=[20, 20], type=list, help='gnn sample')    
    parser.add_argument("--lam1", type=float, default=0.5, help="loss lambda 1")  
    parser.add_argument("--lam2", type=float, default=0.5, help="loss lambda 2")  
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    set_seed(args.seed)
    torch.manual_seed(args.seed) 
    check_path(args.output_dir)
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.data_dir = args.data_dir + args.dataset_name + '/'
    # 准备序列数据
    sequence_data = Sequence_data(args)
    train_id, test_id, valid_id, new_tokens, entity_id_lst = sequence_data.POI_data_pre()
    args.data_file = args.data_dir + args.data_name + '.txt'
    # 获得user_num和所有POI的个数
    user_num, item_num = get_matrix_and_num(args.data_file)
    args.item_size = item_num
    args.user_size = user_num
    # train.pkl:
    # four list: user_id, item_seq of every user_id (move the last 3 item), target_item of every item_seq, seq_len
    train_data = pickle.load(open(args.data_dir + 'train.pkl', 'rb'))
    valid_data = pickle.load(open(args.data_dir + 'valid.pkl', 'rb'))       # 1user - 1seq(remove the last 2 item)
    test_data = pickle.load(open(args.data_dir + 'test.pkl', 'rb'))         # 1user - 1seq(remove the last 1 item)

    # 准备知识图谱数据
    kg = KG_data(args)
    edge_index, edge_type = kg.construct_adj()
    x = torch.arange(0, kg.num_nodes).long().view(-1, 1)            # (num_nodes,1),node编码
    Graph_data = Data(x, edge_index, edge_attr=edge_type)
    args.num_nodes = kg.num_nodes
    args.num_rels = kg.num_rels
    args.num_users = kg.num_users
    args.num_locs = item_num
    global_graph = Graph_data

    # model args str
    args_str = f'{args.save_name}-{args.sample_size}-{args.train_model}'
    # save model
    cur_t = int(time.time())
    checkpoint = args_str + str(cur_t) + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # 不同模型的定义
    if args.train_model == 'GRU':           # GRU(+user)
        model = Simple_GRU(args=args)
    elif args.train_model == 'STKG_PLM':    # full model
        model = STKG_PLM(args=args, global_graph=global_graph, kg=kg, new_tokens=new_tokens, entity_id_lst=entity_id_lst)
    elif args.train_model == 'Bert4PR':     # Bert + user
        model = Bert4PR(args=args, new_tokens=new_tokens, entity_id_lst=entity_id_lst)
    
    # 训练方法类
    trainer = GCL4SR_Train(model, args)
    
    # 构建dataloader和bert prompts
    if args.train_model == 'STKG_PLM' or args.train_model == 'Bert4PR':
        bertdata = MakePrompts(train_id=train_id, test_id=test_id, valid_id=valid_id, datasets_name=args.dataset_name)

        bert_pretrain = bertdata.get_data('train')
        bert_pretrain_test = bertdata.get_data('test')
        bert_pretrain_valid = bertdata.get_data('valid')
        bert_dataset = trainer.model.bert_model.make_dataloader(bert_pretrain)
        bert_dataset_test = trainer.model.bert_model.make_dataloader(bert_pretrain_test)
        bert_dataset_valid = trainer.model.bert_model.make_dataloader(bert_pretrain_valid)

        train_dataset = GCL4SRData(args, train_data)
        myDataset = MyDataset(dataset1=train_dataset, dataset2=bert_dataset)
        train_sampler = torch.utils.data.RandomSampler(myDataset)                                      
        train_dataloader = torch.utils.data.DataLoader(myDataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=False, num_workers=16, drop_last=True)

        eval_dataset = GCL4SRData(args, valid_data)
        myDataset_eval = MyDataset(dataset1=eval_dataset, dataset2=bert_dataset_valid)
        eval_sampler = torch.utils.data.SequentialSampler(myDataset_eval)
        eval_dataloader = torch.utils.data.DataLoader(myDataset_eval, batch_size=args.batch_size, sampler=eval_sampler, drop_last=True)

        test_dataset = GCL4SRData(args, test_data)
        myDataset_test = MyDataset(dataset1=test_dataset, dataset2=bert_dataset_test)
        test_sampler = torch.utils.data.SequentialSampler(myDataset_test)
        test_dataloader = torch.utils.data.DataLoader(myDataset_test, batch_size=args.batch_size, sampler=test_sampler, drop_last=True)
    elif args.train_model == 'GRU':
        train_dataset = GCL4SRData(args, train_data)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, pin_memory=True, num_workers=8)

        eval_dataset = GCL4SRData(args, valid_data)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = GCL4SRData(args, test_data)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    
    # checkpoint
    if args.load_ckpt:
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        scores = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)

    else:
        if args.TransR == True:
            print("==============================[TransR Training...]==============================")
            trans_loss = TransR_train(args, trainer.model, trainer.model.optimizer)
            print("TransR Loss:")
            print(trans_loss)

        print("==============================[" + args.train_model + " Training...]==============================")
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train_stage(epoch, train_dataloader)
            scores = trainer.eval_stage(epoch, eval_dataloader, full_sort=True, test=False)
            early_stopping(np.array(scores[:-2]), trainer.model)
            # early_stopping(np.array(scores), trainer.model)
            # print("test:")
            # scores = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores = trainer.eval_stage(0, test_dataloader, full_sort=True, test=True)
        data = {
                'Metric': ['avg_acc_Top-1', 'avg_acc_Top-5', 'avg_acc_Top-10', 'avg_mrr-10',  'avg_ndcg_Top-5', 'avg_ndcg_Top-10'],
                'Value': scores
                }
        # 创建 DataFrame
        df = pd.DataFrame(data)
        # 保存为 Excel 文件
        df.to_csv('./output/' + args.save_name + '-' + str(args.sample_size) + '-' + str(args.train_model)+ '-result.csv', mode='a', index=False)
    print(args_str)

if __name__ == '__main__':
    main()

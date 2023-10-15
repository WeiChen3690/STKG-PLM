# -*- coding: utf-8 -*-
import numpy as np
import tqdm
import pandas as pd
import torch
import math
from utils import recall_at_k, ndcg_k, get_metric, acc_at_k
import torch.nn.functional as F

class Trainer:
    def __init__(self, model, args):

        self.args = args
        self.model = model
        if self.model.cuda_condition:
            self.model.to(self.model.device)

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        ACC_1, NDCG_1, MRR = get_metric(pred_list, 1)
        ACC_5, NDCG_5, MRR = get_metric(pred_list, 5)
        ACC_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "ACC@1": '{:.4f}'.format(ACC_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "ACC@5": '{:.4f}'.format(ACC_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "ACC@10": '{:.4f}'.format(ACC_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [ACC_1, NDCG_1, ACC_5, NDCG_5, ACC_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        acc, ndcg = [], []
        for k in [1, 5, 10]:
            acc.append(acc_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "ACC@1": '{:.4f}'.format(acc[0]), "NDCG@1": '{:.4f}'.format(ndcg[0]),
            "ACC@5": '{:.4f}'.format(acc[1]), "NDCG@5": '{:.4f}'.format(ndcg[1]),
            "ACC@10": '{:.4f}'.format(acc[2]), "NDCG@10": '{:.4f}'.format(ndcg[2])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [acc[0], ndcg[0], acc[1], ndcg[1], acc[2], ndcg[2]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.model.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.node_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.node_embeddings.weight[:self.args.num_locs]
        # [batch, hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        # rating_pred = F.log_softmax(rating_pred, dim=-1)
        return rating_pred


class GCL4SR_Train(Trainer):
    def __init__(self, model, args):
        super(GCL4SR_Train, self).__init__(
            model,                                                 # GCL4SR
            args
        )

    def get_acc(self, target, scores):
        """target and scores are torch cuda Variable"""
        target = target.data.cpu().numpy()
        val, idxx = scores.data.topk(10, 0)
        p = idxx.cpu().numpy()
        acc = np.zeros((3, 1))
        # for i, p in enumerate(predx):
        t = target
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
        return acc

    def get_ndcg(self, target, scores):
        """target and scores are torch cuda Variable"""
        target = target.data.cpu().numpy()
        val, idxx = scores.data.topk(10, 0)
        p = idxx.cpu().numpy()
        # acc = np.zeros((3, 1))
        ndcg = np.zeros((3, 1))
        # for i, p in enumerate(predx):
        t = target
        if t != 0:
            if t in p[:10] and t > 0:
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        return ndcg

    def train_stage(self, epoch, train_dataloader):

        desc = f'n_sample-{self.args.sample_size}-' \
               f'hidden_size-{self.args.hidden_size}'
        # 开始训练
        train_data_iter = tqdm.tqdm(enumerate(train_dataloader),
                                       desc=f"{self.args.train_model}-{self.args.dataset_name} Epoch:{epoch}",
                                       total=len(train_dataloader),
                                       bar_format="{l_bar}{r_bar}")
        if self.args.train_model != 'GRU':
            self.model.bert_model.train()
        self.model.train()
        joint_loss_avg = 0.0
        main_loss_avg = 0.0
        cl_loss_avg = 0.0
        mmd_loss_avg = 0.0

        for i, batch in train_data_iter:                                                # batch的0,1,2分别是，uid, seq, target
            # 0. batch_data will be sent into the device(GPU or CPU)
            if self.args.train_model != 'GRU':
                bert_batch = batch[1]
                batch = tuple(t.to(self.model.device) for t in batch[0])
            else:
                batch = tuple(t.to(self.model.device) for t in batch)

            if self.args.train_model == 'GCL4SR':
                joint_loss, main_loss, cl_loss, mmd_loss = self.model.train_stage(batch)
                self.model.optimizer.zero_grad()
                joint_loss.backward()
                self.model.optimizer.step()
                joint_loss_avg += joint_loss.item()
                main_loss_avg += main_loss.item()
                cl_loss_avg += cl_loss.item()
                mmd_loss_avg += mmd_loss.item()
            elif self.args.train_model == 'GRU':
                joint_loss, _ = self.model.train_stage(batch)
                self.model.optimizer.zero_grad()
                joint_loss.backward()
                self.model.optimizer.step()
                joint_loss_avg += joint_loss.item()
            elif self.args.train_model == 'STKG_PLM':
                joint_loss, main_loss, cl_loss = self.model.train_stage(batch, bert_batch)
                self.model.optimizer.zero_grad()
                joint_loss.backward()
                self.model.optimizer.step()
                joint_loss_avg += joint_loss.item()
                main_loss_avg += main_loss.item()
                cl_loss_avg += cl_loss.item()
            elif self.args.train_model == 'Bert4PR':
                joint_loss = self.model.train_stage(batch, bert_batch)
                self.model.optimizer.zero_grad()
                joint_loss.backward()
                self.model.optimizer.step()
                joint_loss_avg += joint_loss.item()
                main_loss_avg += 0
                cl_loss_avg += 0
                mmd_loss_avg += 0
            
        self.model.scheduler.step()
        post_fix = {
            "joint_loss_avg": '{:.4f}'.format(joint_loss_avg / len(train_data_iter)),
            "main_loss_avg": '{:.4f}'.format(main_loss_avg / len(train_data_iter)),
            "gcl_loss_avg": '{:.4f}'.format(cl_loss_avg / len(train_data_iter))
        }
        print(desc)
        print(str(post_fix))
    
    def eval_stage(self, epoch, dataloader, full_sort=False, test=True):
        users_acc = {}
        users_ndcg = {}
        mrr = {}
        str_code = "test" if test else "eval"
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        self.model.eval()
        if self.args.train_model != 'GRU':
            self.model.bert_model.eval()
        pred_list = None

        if full_sort:
            answer_list = None
            users_rnn_acc_top1 = {}
            users_rnn_acc_top5 = {}
            users_rnn_acc_top10 = {}
            users_rnn_ndcg_top1 = {}
            users_rnn_ndcg_top5 = {}
            users_rnn_ndcg_top10 = {}
            users_mrr = {}
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                if self.args.train_model != 'GRU':
                    bert_batch = batch[1]
                    batch = tuple(t.to(self.model.device) for t in batch[0])
                else:
                    batch = tuple(t.to(self.model.device) for t in batch)
                user_ids = batch[0]
                answers = batch[2]
                if self.args.train_model == 'GCL4SR' or self.args.train_model == 'STKG_PLM':
                    recommend_output = self.model.eval_stage(batch, bert_batch)
                    rating_pred = self.predict_full(recommend_output)       # (batch_size, num_poi)
                elif self.args.train_model == 'Bert4PR':
                    recommend_output = self.model.eval_stage(batch, bert_batch)   
                    rating_pred = self.predict_full(recommend_output)  
                elif self.args.train_model == 'GRU':
                    _, rating_pred = self.model.train_stage(batch)
                for u in user_ids.cpu().numpy().tolist():
                    users_acc[u] = [0, 0, 0, 0]
                    users_ndcg[u] = [0, 0, 0, 0]
                    mrr[u] = [0, 0]
                for index, u in enumerate(user_ids.cpu().numpy().tolist()):
                    users_ndcg[u][0] += len(answers.unsqueeze(1)[index])
                    users_acc[u][0] += len(answers.unsqueeze(1)[index])
                    acc = self.get_acc(answers.unsqueeze(1)[index], rating_pred[index])
                    ndcg = self.get_ndcg(answers.unsqueeze(1)[index], rating_pred[index])
                    sorted_rating_pred = rating_pred[index].sort(descending=True).indices
                    for i in range(len(sorted_rating_pred)):
                        if i >= 10:
                            mrr[u][1] += np.array(0.0)
                            break
                        if sorted_rating_pred[i] == answers.unsqueeze(1)[index]:
                            rank = i + 1
                            mrr[u][1] += np.array(1 / rank)
                            break
                    mrr[u][0] += len(answers.unsqueeze(1)[index])
                    # Top-1
                    users_acc[u][1] += acc[2]
                    users_ndcg[u][1] += ndcg[2]
                    # Top-5
                    users_acc[u][2] += acc[1]
                    users_ndcg[u][2] += ndcg[1]
                    # top-10
                    users_acc[u][3] += acc[0]
                    users_ndcg[u][3] += ndcg[0]
            
            for u in users_acc:
                # Top-1
                tmp_acc = users_acc[u][1] / users_acc[u][0]
                users_rnn_acc_top1[u] = tmp_acc.tolist()[0]
                # Top-5
                tmp_acc = users_acc[u][2] / users_acc[u][0]
                users_rnn_acc_top5[u] = tmp_acc.tolist()[0]
                # Top-10
                tmp_acc = users_acc[u][3] / users_acc[u][0]
                users_rnn_acc_top10[u] = tmp_acc.tolist()[0]

                tmp_mrr = np.array(mrr[u][1] / mrr[u][0])
                users_mrr[u] = tmp_mrr.tolist()

            avg_acc_top1 = np.mean([users_rnn_acc_top1[x] for x in users_rnn_acc_top1])
            avg_acc_top5 = np.mean([users_rnn_acc_top5[x] for x in users_rnn_acc_top5])
            avg_acc_top10 = np.mean([users_rnn_acc_top10[x] for x in users_rnn_acc_top10])
            avg_mrr = np.mean([users_mrr[x] for x in users_mrr])
        
            print('avg_acc_Top-1:{}'.format(avg_acc_top1))
            print('avg_acc_Top-5:{}'.format(avg_acc_top5))
            print('avg_acc_Top-10:{}'.format(avg_acc_top10))
            print('avg_mrr-10:{}'.format(avg_mrr))
            for u in users_ndcg:
                # Top-1
                tmp_ndcg = users_ndcg[u][1] / users_ndcg[u][0]
                users_rnn_ndcg_top1[u] = tmp_ndcg.tolist()[0]
                # Top-5
                tmp_ndcg = users_ndcg[u][2] / users_ndcg[u][0]
                users_rnn_ndcg_top5[u] = tmp_ndcg.tolist()[0]
                # Top-10
                tmp_ndcg = users_ndcg[u][3] / users_ndcg[u][0]
                users_rnn_ndcg_top10[u] = tmp_ndcg.tolist()[0]

            avg_ndcg_top1 = np.mean([users_rnn_ndcg_top1[x] for x in users_rnn_ndcg_top1])
            avg_ndcg_top5 = np.mean([users_rnn_ndcg_top5[x] for x in users_rnn_ndcg_top5])
            avg_ndcg_top10 = np.mean([users_rnn_ndcg_top10[x] for x in users_rnn_ndcg_top10])

            print('avg_ndcg_Top-1:{}'.format(avg_ndcg_top1))
            print('avg_ndcg_Top-5:{}'.format(avg_ndcg_top5))
            print('avg_ndcg_Top-10:{}'.format(avg_ndcg_top10))
            
            return [avg_acc_top1, avg_ndcg_top1, avg_acc_top5,  avg_mrr,  avg_ndcg_top5, avg_acc_top10, avg_ndcg_top10]

# -*- coding: utf-8 -*-
import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn.init import xavier_normal_, constant_
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GCNConv, GATConv
import transformers
from transformers import BertTokenizerFast, BertModel, BertConfig, BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer
from torch.nn import Module
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from dataset import PreBertDataset
from tqdm import tqdm

from mk_data import KG_data


class PreBert(Module):
    def __init__(self, args):
        super(PreBert, self).__init__()
        self.args = args
        self.special = {'additional_special_tokens': ['[TRA]', '[CHE]', '[STR]', '[SHT]', '[MID]', '[LNG]']}
        # load 分词器
        self.tokenizer = AutoTokenizer.from_pretrained('./bert', do_lower_case=False)
        # load model
        self.model = AutoModelForMaskedLM.from_pretrained('./bert')
        self.linear = nn.Linear(self.args.hidden_size, 312)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
    
    def init_tokenizer(self, new_tokens):
        self.token_sum = len(new_tokens)
        
        self.tokenizer.add_special_tokens(self.special)

        self.tokenizer.add_tokens(new_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.save_pretrained('./bert')
        self.tokenizer = AutoTokenizer.from_pretrained('./bert', do_lower_case=False)
    
    def init_new_token(self, new_tokens, entity_id_lst, node_embeddings):
        for entity_id in entity_id_lst:
            # hidden_size = self.model.bert.embeddings.word_embeddings.weight.shape[1]
            # entity_embedding = torch.randn((hidden_size))
            entity_embedding = self.linear(node_embeddings(torch.tensor(entity_id).to(torch.device("cuda:{}".format(self.args.gpu_id)))))
            entity_embedding = self.dropout(entity_embedding)
            self.model.bert.embeddings.word_embeddings.weight.data[-1 * len(new_tokens) + entity_id] = entity_embedding
    
    def init_new_token_split(self, new_tokens, entity_id_lst, loc_embeddings, user_embeddings, num_locs):
        for entity_id in entity_id_lst:
            # hidden_size = self.model.bert.embeddings.word_embeddings.weight.shape[1]
            # entity_embedding = torch.randn((hidden_size))
            if entity_id <= num_locs:
                entity_embedding = self.linear(loc_embeddings(torch.tensor(entity_id).to(torch.device("cuda:{}".format(self.args.gpu_id)))))
            else:
                entity_embedding = self.linear(user_embeddings(torch.tensor(entity_id - num_locs).to(torch.device("cuda:{}".format(self.args.gpu_id)))))
            entity_embedding = self.dropout(entity_embedding)
            self.model.bert.embeddings.word_embeddings.weight.data[-1 * len(new_tokens) + entity_id] = entity_embedding
    
    def make_dataloader(self, bert_data):
        bert_dataset = PreBertDataset(self.tokenizer, 256, bert_data)
        return bert_dataset

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        return outputs

    def result(self, tail_mask, attention_mask, tail_index, text, tail_pos, labels):
        outputs = self.model(tail_mask, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        tail_logits = torch.zeros((logits.shape[0], logits.shape[2])).cuda()            # [16, 40458]
        for i in range(logits.shape[0]):
            tail_logits[i, :] = logits[i, tail_pos[i, 0], :]
        return tail_logits[:, -1 * self.token_sum:]


class KGAT(Module):
    def __init__(self, args, in_features, out_features):
        super(KGAT, self).__init__()
        self.num_layers = len(args.sample_size)
        self.dropout = 0.4
        self.alpha = 0.2
        self.GAT_Layer = GraphAttentionLayer(args, in_features, out_features, dropout=self.dropout, alpha=self.alpha, concat=False)
    
    def forward_relation(self, src_embs, dst_embs, rel_embs, adj, padding_mask):
        for i, (edge_index, rel_idx, size) in enumerate(adj):                             # 不用子图编码edge_index
            if len(list(src_embs.shape)) < 2:
                src_embs = src_embs.unsqueeze(0)
            rel_embs = F.dropout(rel_embs, self.dropout, training=self.training)          # (num_heads, num_nei, hidden_size)
            src_embs = F.dropout(src_embs, self.dropout, training=self.training)          # src embedding
            dst_embs = F.dropout(dst_embs, self.dropout, training=self.training)          # tar embedding
            x = self.GAT_Layer.forward_relation(src_embs, dst_embs, rel_embs, padding_mask)
            x = self.GAT_Layer.forward_relation(x, dst_embs, rel_embs, padding_mask)
            x = F.dropout(x, self.dropout, training=self.training)
            
            x = x[:size[1]]                                                           # 取出需要的src节点
            y = dst_embs[:size[1]]                                                    # 取出需要的dst节点
            z = rel_embs[:size[1]]
            if i != self.num_layers - 1:                                              # 352->1484->4306
                src_embs = x
                dst_embs = y
                rel_embs = z
                padding_mask = padding_mask[:size[1]]
        return x
    

class GraphAttentionLayer(Module):                                              # used in KGAT
    def __init__(self, args, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.args = args
        self.device = torch.device("cuda:{}".format(self.args.gpu_id))

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))    # (in_fea, out_fea)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))          # (2 * out_fea, 1)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.fc = nn.Linear(2 * out_features, out_features)                       # (2 * out_fea, out_fea)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward_relation(self, src_embs, dst_embs, relations, adj):
        # src_embs:   (src_num, dim)
        # dst_embs:   (src_num, dst_num, dim)
        # relations:  (src_num, dst_num, dim)
        # adj:        (src_num, dst_num) padding mask
        # dst_num为所有出现过的尾实体，方便对齐

        # (src_num, dst_num, dim),一个src对应多个dst
        src = src_embs.unsqueeze(1).expand(dst_embs.size())
        # (src_num, dst_num, dim)
        dst = dst_embs
        a_input = torch.cat((src, dst),dim=-1)                                 # (src_num, dst_num, 2*dim)
        # 计算相应的weight
        
        # (src_num, dst_num, dim)  (src_num, dst_num, dim),
        e_input = torch.cosine_similarity(self.fc(a_input), relations, dim=-1).squeeze()
        e = self.leakyrelu(e_input)                                            # (src_num, dst_num)

        zero_vec = -9e15*torch.ones_like(e).to(self.device)
        adj = adj.to(self.device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # (src_num, dst_num)
        # (src_num, 1, e_num) * (src_num, e_num, out_features) -> (src_num, out_features)
        entity_emb_weighted = torch.bmm(attention.unsqueeze(1), dst_embs).squeeze()
        # 加上自己的特征
        h_prime = entity_emb_weighted + src_embs

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class Simple_GRU(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # used for simple GRU
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id))
        self.node_embeddings = nn.Embedding(args.num_nodes + 1, args.hidden_size)

        self.GRU_encoder = nn.GRU(1 * args.hidden_size, args.hidden_size)
        self.fc = nn.Linear(args.hidden_size, args.item_size)
        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    
    def train_stage(self, data):
        targets = data[2]
        seq = data[1]
        user = data[0]
        seq_length = data[3]

        seq_flat = seq.flatten()
        seq_mask = (seq_flat == 0).float().unsqueeze(-1)                       # (batch_size, 10 ,1)
        seq_mask = 1.0 - seq_mask                                        # padding的地方为0,有效的地方为1
        
        h1 = Variable(torch.zeros(1, self.args.batch_size, self.args.hidden_size)).to(self.device)

        user_emb = self.node_embeddings(user)
        user_emb = user_emb.reshape(seq.shape[0], 1, -1)
        user_emb_x = user_emb.repeat(1, seq.shape[1], 1)

        loc_emb = self.node_embeddings(seq)
        x = loc_emb.permute(1, 0, 2)

        pack_x = nn.utils.rnn.pack_padded_sequence(x, lengths=seq_length.cpu(), enforce_sorted=False)
        hidden_state, h1 = self.GRU_encoder(pack_x, h1)
        out = nn.utils.rnn.pad_packed_sequence(hidden_state, batch_first=True)[0]                   # [batch_size, 10, hidden_size]
        origin_len = seq_length                                                                     # [batch_size]
        final_out_index = torch.tensor(origin_len) - 1
        final_out_index = final_out_index.reshape(final_out_index.shape[0], 1, -1)                  # [batch_size, 1, 1]
        final_out_index = final_out_index.repeat(1, 1, self.args.hidden_size).to(self.device)       # [batch_size, 1, hidden_size]
        out = torch.gather(out, 1, final_out_index).squeeze(1)                                      # [batch_size, hidden_size]
        out = F.selu(out)                   # forward 里面的hidden
        Y = self.fc(out)
        score = F.log_softmax(Y)
        joint_loss = self.criterion(score, targets)       
        return joint_loss, score      
    

class STKG_PLM(nn.Module):
    def __init__(self, args, global_graph, kg, new_tokens, entity_id_lst) -> None:
        super().__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")
        self.kg = kg

        self.node_embeddings = nn.Embedding(args.num_nodes + 1, args.hidden_size).to(self.device)
        self.relation_embeddings = nn.Embedding(args.num_rels + 2, args.hidden_size)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.global_graph = global_graph.to(self.device)

        # bert
        self.bert_model = PreBert(args)
        self.bert_model.to(torch.device("cuda:{}".format(args.gpu_id)))
        self.bert_model.init_tokenizer(new_tokens)
        self.bert_model.init_new_token(new_tokens, entity_id_lst, self.node_embeddings)
        self.bert_linear = nn.Linear(312, self.args.hidden_size)

        # kg
        self.global_gat = KGAT(args, args.hidden_size, args.hidden_size)
        self.head_num = args.num_nodes
        self.kg_dict, self.kg_dict_t, self.head2rel = self.kg.sample_align(self.head_num)

        # sequence encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1 * args.hidden_size,
                                                        nhead=args.num_attention_heads,
                                                        dim_feedforward=4 * args.hidden_size,
                                                        dropout=args.attention_probs_dropout_prob,
                                                        activation=args.hidden_act)
        self.item_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=args.num_hidden_layers)

        # time encoder
        self.trans_encoder_layer = nn.TransformerEncoderLayer(d_model=1 * args.hidden_size,
                                                        nhead=args.num_attention_heads,
                                                        dim_feedforward=4 * args.hidden_size,
                                                        dropout=args.attention_probs_dropout_prob,
                                                        activation=args.hidden_act)
        self.trans_encoder = nn.TransformerEncoder(self.trans_encoder_layer, num_layers=args.num_hidden_layers)

        # AttNet
        self.w_1 = nn.Parameter(torch.Tensor(3 * args.hidden_size, 2 * args.hidden_size))
        self.w_2 = nn.Parameter(torch.Tensor(args.hidden_size, 1))
        self.w_3 = nn.Linear(2 * args.hidden_size, args.hidden_size)

        self.final_lamda = nn.Parameter(torch.rand(1, 3))
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear_final = nn.Linear(6 * args.hidden_size, 1 * args.hidden_size)

        self.gnndrop = nn.Dropout(args.gnn_dropout_prob)
        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def _L2_loss_mean(self, x):
        return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

    def calc_kg_loss_transE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embeddings(r)          # (kg_batch_size, relation_dim)
        # (kg_batch_size, entity_dim)
        h_embed = self.node_embeddings(h)
        pos_t_embed = self.node_embeddings(pos_t)      # (kg_batch_size, entity_dim)
        neg_t_embed = self.node_embeddings(neg_t)      # (kg_batch_size, entity_dim)
        # Equation (1)
        pos_score = torch.sum(torch.pow(h_embed + r_embed - pos_t_embed, 2), dim=1)     # (kg_batch_size)
        neg_score = torch.sum(torch.pow(h_embed + r_embed - neg_t_embed, 2), dim=1)     # (kg_batch_size)
        # Equation (2)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = self._L2_loss_mean(h_embed) + self._L2_loss_mean(r_embed) + \
            self._L2_loss_mean(pos_t_embed) + self._L2_loss_mean(neg_t_embed)
        # # TODO: optimize L2 weight
        loss = kg_loss + 1e-3 * l2_loss
        return loss
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def gat_encode(self, items):                                                      # GNN, items: seq\
        # sizes: 每一层需要采样的邻居数目，一共两层，每层20个neighbor，-1则选取所有邻居
        # 先采样出子图，返回邻居子图
        subgraph_loaders = NeighborSampler(self.global_graph.edge_index, node_idx=items, sizes=self.args.sample_size,
                                            shuffle=False,
                                            num_workers=0, batch_size=items.shape[0])       # 对大规模kg进行小批量训练  [20, 20]
        g_adjs = []
        src_nodes = []                                                                      # 子图shape: (14657, 10784),88953个边
        for (batch_size, node_idx, adjs) in subgraph_loaders:                               # adjs:1-n层采样结果的list
            if type(adjs) == list:                                                          # node_idx,L层采样中遇到的所有节点的list,target在最前端
                g_adjs = [adj.to(items.device) for adj in adjs]
            else:
                g_adjs = adjs.to(items.device)
            dst_nodes = torch.tensor([self.kg_dict[x] for x in node_idx.tolist()], dtype=torch.long).to(self.device)                      
            rel = torch.tensor([self.head2rel[x] for x in node_idx.tolist()], dtype=torch.long).to(self.device)                                         
            n_idxs = node_idx.to(items.device)                                              # n_idx是本次所采样到的所有节点id
            src_nodes = self.node_embeddings(n_idxs).squeeze()                              # (num_nodes, item_embedding), 采样到所有节点的表征
        dst_nodes = self.node_embeddings(dst_nodes).squeeze()                               # (num_nodes, num_nei, item_embedding), 采样到所有节点的所有neighbor的表征
        rel_embs = self.relation_embeddings(rel).squeeze()                                  # (num_nodes, num_nei, rel_embedding), 采样到所有节点与之所有neighbor之间的relation的表征     
        item_entities = torch.stack([self.kg_dict_t[x] for x in node_idx.tolist()])         # dst_nodes    
        padding_mask = torch.where(item_entities != 0, torch.ones_like(item_entities), torch.zeros_like(item_entities)).float()     # 为dst_nodes的padding与否生成mask,因为默认neighbors数量相同
        g_hidden = self.global_gat.forward_relation(src_nodes, dst_nodes, rel_embs, g_adjs, padding_mask)         # KGAT
        return g_hidden

    def final_att_net(self, seq_mask, hidden):                                     # Attention(权重矩阵在前面已经乘过了)
        batch_size = hidden.shape[0]
        lens = hidden.shape[1]
        pos_emb = self.position_embeddings.weight[:lens]                                    # (10, hidden_size) 在batch维度上进行repeat
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)                             # (batch_size, 10, hidden_size)
        # POI的平均表征->seq的表征, 平均池化
        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)              # batch的每个seq的poi表示加起来 / batch的每个seq中不是padding的数据总数 (batch_size, hidden_size)
        seq_hidden = seq_hidden.unsqueeze(-2).repeat(1, lens, 1)                            # (batch_size, 10, 2 * hidden_size)
        
        # 具体item的表征,将位置信息融入
        item_hidden = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)              # position embedding 与 seq embedding拼接
        item_hidden = torch.tanh(item_hidden)                                               
        # 计算每一个位置的score
        att_score = torch.cosine_similarity(item_hidden, seq_hidden, dim=2)                   # (batch_size, 10)
        att_score = att_score.unsqueeze(-1)
        
        att_score_masked = att_score * seq_mask                                             # 每一个序列中每一个poi的权重
        output = torch.sum(att_score_masked * hidden, 1)                                    # (batch_size, 2 * hidden_size)
        output = self.w_3(output)               
        return output
    
    def mean_pooling(self, seq_mask, hidden):                                  
        # POI的平均表征->seq的表征, 平均池化
        seq_hidden = torch.sum(hidden * seq_mask, -2) / torch.sum(seq_mask, 1)              # batch的每个seq的poi表示加起来 / batch的每个seq中不是padding的数据总数 (batch_size, hidden_size)
        return seq_hidden

    def GCL_loss(self, hidden, hidden_norm=True, temperature=1.0):
        batch_size = hidden.shape[0] // 2
        LARGE_NUM = 1e9
        # inner dot or cosine
        if hidden_norm:
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        hidden_list = torch.split(hidden, batch_size, dim=0)
        hidden1, hidden2 = hidden_list[0], hidden_list[1]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = torch.from_numpy(np.arange(batch_size)).to(hidden.device)
        masks = torch.nn.functional.one_hot(torch.from_numpy(np.arange(batch_size)).to(hidden.device), batch_size)

        logits_aa = torch.matmul(hidden1, hidden1_large.transpose(1, 0)) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.transpose(1, 0)) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.transpose(1, 0)) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.transpose(1, 0)) / temperature

        loss_a = torch.nn.functional.cross_entropy(torch.cat([logits_ab, logits_aa], 1), labels)
        loss_b = torch.nn.functional.cross_entropy(torch.cat([logits_ba, logits_bb], 1), labels)
        loss = (loss_a + loss_b)
        return loss

    def generate_square_subsequent_mask(self, sz: int) -> Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, -10000.0).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, data, bert_data, mode):
        user_ids = data[0]                                               # uid(batch_size)
        seq = data[1]                                                    # seq
        sequence_len = data[3]
        trans_seq = data[4]                                              # trans_seq

        seq_flat = seq.flatten()
        seq_mask = (seq == 0).float().unsqueeze(-1)                      # (batch_size, 10 ,1)
        seq_mask = 1.0 - seq_mask                                        # padding的地方为0,有效的地方为1

        # bert model
        input_ids = bert_data['input_ids'].to(self.device)             
        attention_mask = bert_data['attention_mask'].to(self.device)
        labels = bert_data['labels'].to(self.device)                    # 
        tail_pos = np.array(bert_data['tail_pos'].squeeze()).tolist()
        if mode == 'train':                                             # 随机mask训练
            result = self.bert_model(input_ids, attention_mask, labels)
            bert_hidden = result.hidden_states[-1]
            bert_seq_hidden = torch.zeros((self.args.batch_size, 312)).to(self.device)      # [batch_size, bert_hidden_size]
            for i in range(self.args.batch_size):
                bert_seq_hidden[i,:] = bert_hidden[i, tail_pos[i], :]
            bert_seq_hidden = self.bert_linear(bert_seq_hidden)
        elif mode == 'eval':                                            # 只mask tail POI预测
            tail_mask = bert_data['tail_mask'].to(self.device)
            result = self.bert_model(tail_mask, attention_mask, labels)
            tail_pos = np.array(bert_data['tail_pos'].squeeze()).tolist()
            bert_hidden = result.hidden_states[-1]
            bert_seq_hidden = torch.zeros((self.args.batch_size, 312)).to(self.device)
            for i in range(self.args.batch_size):
                bert_seq_hidden[i,:] = bert_hidden[i, tail_pos[i], :]
            bert_seq_hidden = self.bert_linear(bert_seq_hidden)

        # KGAT
        # 加入user node, category因为在kg_dict的构造中就已经放入了
        node_flat = torch.tensor(seq_flat.tolist() + user_ids.tolist(), dtype=torch.long).to(self.device)
        seq_hidden_augmented_a = self.gat_encode(node_flat)[:seq_flat.shape[0]].view(-1, self.args.max_seq_length, self.args.hidden_size)  # augmented a
        user_hidden_augmented_a = self.gat_encode(node_flat)[seq_flat.shape[0]:]
        seq_hidden_augmented_b = self.gat_encode(node_flat)[:seq_flat.shape[0]].view(-1, self.args.max_seq_length, self.args.hidden_size) # (batch_size, 10, hidden_size)
        user_hidden_augmented_b = self.gat_encode(node_flat)[seq_flat.shape[0]:]
        seq_hidden_augmented = (seq_hidden_augmented_a + seq_hidden_augmented_b) / 2
        # KGAT后的user embedding
        user_hidden_augmented = (user_hidden_augmented_a + user_hidden_augmented_b) / 2              # [batch_size, hidden_size]
        user_hidden_augmented_z = user_hidden_augmented.reshape(seq.shape[0], 1, -1)              # [batch_size, 1, hidden_size]
        user_hidden_augmented_x = user_hidden_augmented_z.repeat(1, seq.shape[1], 1)              # [batch_size, 10, hidden_size]
        user_hidden_augmented_trans = user_hidden_augmented_z.repeat(1, seq.shape[1] - 1, 1)    

        # trans embedding
        trans_emb = self.relation_embeddings(trans_seq)
        trans_emb = self.dropout(trans_emb)
        trans_emb[:, 0, :] = user_hidden_augmented
        trans_emb[:, 1:, ] = trans_emb[:, 1:, ] + user_hidden_augmented_trans                  # [batch_size, seq_len, hidden_size]

        # loc emb
        key_padding_mask = (seq == 0)
        attn_mask = self.generate_square_subsequent_mask(self.args.max_seq_length).to(seq.device)
        seq_hidden_local = self.node_embeddings(seq)                        # (batch_size, seq_len, hidden_size)
        seq_hidden_local = self.dropout(seq_hidden_local)
        seq_hidden_local = seq_hidden_local + user_hidden_augmented_x          # 加入用户信息
        seq_hidden_permute = seq_hidden_local.permute(1, 0, 2)              # (seq_len, batch_size, hidden_size)

        # loc seq transformer encoder
        encoded_layers = self.item_encoder(seq_hidden_permute,                                                                            
                                            mask=attn_mask,
                                            src_key_padding_mask=key_padding_mask)
        sequence_output = encoded_layers.permute(1, 0, 2)                   # (batch_size, seq_len, hidden_size)
        out_index = torch.tensor(sequence_len) - 1
        out_index = out_index.reshape(out_index.shape[0], 1, -1)
        out_index = out_index.repeat(1, 1, self.args.hidden_size).to(self.device)
        sequence_output = torch.gather(sequence_output, 1, out_index).squeeze(1)

        # trans transformer encoder
        trans_permute = trans_emb.permute(1, 0, 2)
        encoded_layers_trans = self.trans_encoder(trans_permute,                                                                    
                                            mask=attn_mask,
                                            src_key_padding_mask=key_padding_mask)
        trans_output = encoded_layers_trans.permute(1, 0, 2)                            # (batch_size, seq_len, hidden_size)
        out_index = torch.tensor(sequence_len) - 1
        out_index = out_index.reshape(out_index.shape[0], 1, -1)
        out_index = out_index.repeat(1, 1, self.args.hidden_size).to(self.device)
        trans_output = torch.gather(trans_output, 1, out_index).squeeze(1)              # [batch_size, hidden_size]

        # 将seq和trans的结果取平均
        seqtrans_output = (sequence_output + trans_output) / 2

        # 输入self-attn的图表征,加入用户信息指导
        seq_hidden_augmented_user = torch.cat([seq_hidden_augmented, user_hidden_augmented_x], -1)       # [batch_size, max_len, 2 * hidden_size]
        # 给bert加入用户信息,为了保证数据维度所以直接相加
        bert_seq_hidden = bert_seq_hidden + user_hidden_augmented                        # [batch_size, hidden_size]

        return (seq_hidden_augmented_a, seq_hidden_augmented_b), seq_mask, seq_hidden_augmented_user, seqtrans_output, bert_seq_hidden

    def train_stage(self, data, bert_data):
        targets = data[2]
        # hidden:           cat完之后要输入predict layer的图表征              (batch_size, 10, hidden_size)
        # seq_gnn_a/b:      图增强表征                                      (batch_size, 10, hidden_size)
        # seq_mask:         padding的地方为0,有效的地方为1                    (batch_size, 10 ,1)
        (seq_gnn_a, seq_gnn_b), seq_mask, seq_hidden_augmented_user, seqtrans_output, bert_seq_hidden = self.forward(data, bert_data, 'train')                      
        # self-attention
        seq_graph_output = self.final_att_net(seq_mask, seq_hidden_augmented_user)                                      # 改为用transformer的hidden预测
        # cat
        self.weights = torch.softmax(self.final_lamda, dim=1)
        weighted_seq_out = self.weights[:, 0] * seqtrans_output + \
                        self.weights[:, 1] * seq_graph_output + \
                        self.weights[:, 2] * bert_seq_hidden
        weighted_seq_out = self.dropout(weighted_seq_out)                                                     # (batch_size, hidden_size)
        # 每一个poi的嵌入表示
        test_item_emb = self.node_embeddings.weight[:self.args.num_locs]                   # (num_poi, hidden_size)
        # calculate main loss, 寻找num_locs个poi中与当前seq表征最相似的那个
        logits = torch.matmul(weighted_seq_out, test_item_emb.transpose(0, 1))                       # (batch_size, num_poi)
        main_loss = self.criterion(logits, targets)
        # GCL
        sum_a = torch.sum(seq_gnn_a * seq_mask, 1) / torch.sum(seq_mask.float(), 1)         # 序列的平均表征
        sum_b = torch.sum(seq_gnn_b * seq_mask, 1) / torch.sum(seq_mask.float(), 1) 
        info_hidden = torch.cat([sum_a, sum_b], 0)
        gcl_loss_loc = self.GCL_loss(info_hidden, hidden_norm=True, temperature=0.5)
        gcl_loss = gcl_loss_loc
        # loss
        joint_loss = self.args.lam1 * main_loss + self.args.lam2 * gcl_loss

        return joint_loss, main_loss, gcl_loss            
        
    def eval_stage(self, data, bert_batch):
        _, seq_mask, seq_hidden_augmented_user, seqtrans_output, bert_seq_hidden = self.forward(data, bert_batch, 'eval')
        seq_graph_output = self.final_att_net(seq_mask, seq_hidden_augmented_user)                                      # 改为用transformer的hidden预测
        self.weights = torch.softmax(self.final_lamda, dim=1)
        weighted_seq_out = self.weights[:, 0] * seqtrans_output + \
                        self.weights[:, 1] * seq_graph_output + \
                        self.weights[:, 2] * bert_seq_hidden
        return weighted_seq_out


# bert-only and bert+user
class Bert4PR(nn.Module):
    def __init__(self, args, new_tokens, entity_id_lst) -> None:
        super().__init__()
        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda:{}".format(self.args.gpu_id) if self.cuda_condition else "cpu")

        self.user_embeddings = nn.Embedding(args.num_users, args.hidden_size).to(self.device)
        self.loc_embeddings = nn.Embedding(args.num_locs + 1, args.hidden_size).to(self.device)
        self.node_embeddings = self.loc_embeddings
        # bert
        self.bert_model = PreBert(args)
        self.bert_model.to(torch.device("cuda:{}".format(args.gpu_id)))
        self.bert_model.init_tokenizer(new_tokens)
        self.bert_model.init_new_token_split(new_tokens, entity_id_lst, self.loc_embeddings, self.user_embeddings, args.num_locs)
        self.bert_linear = nn.Linear(312, self.args.hidden_size)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.linear_final = nn.Linear(2 * args.hidden_size, 1 * args.hidden_size)
        
        self.criterion = nn.CrossEntropyLoss()
        self.betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=self.betas, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        stdv = 1.0 / math.sqrt(self.args.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, data, bert_data):
        user_ids = data[0]                                              
        # bert model
        # random mask
        input_ids = bert_data['input_ids'].to(self.device)
        # tail_mask = bert_data['tail_mask'].to(self.device)
        attention_mask = bert_data['attention_mask'].to(self.device)
        labels = bert_data['labels'].to(self.device)
        tail_pos = np.array(bert_data['tail_pos'].squeeze()).tolist()
        result = self.bert_model(input_ids, attention_mask, labels)
        bert_hidden = result.hidden_states[-1]
        bert_seq_hidden = torch.zeros((self.args.batch_size, 312)).to(self.device)
        for i in range(self.args.batch_size):
            bert_seq_hidden[i,:] = bert_hidden[i, tail_pos[i], :]
        bert_seq_hidden = self.bert_linear(bert_seq_hidden)
        # 训练时用随机mask，直接使用bert的loss训练
        bert_loss = result.loss                                

        # user embedding
        user_emb = self.user_embeddings(user_ids - self.args.num_locs)
        user_emb = self.dropout(user_emb)
        user_emb_gate = user_emb.view(-1, self.args.hidden_size)           # [batch_size, hidden_size]

        bert_seq_hidden = torch.cat([bert_seq_hidden, user_emb_gate], dim=-1)
        return bert_seq_hidden, bert_loss
    
    def forward_eval(self, data, bert_data):
        user_ids = data[0]                                               # uid(batch_size)
        # 验证时使用tail_mask，只对最后一个POI token进行mask并进行预测，logits是得到的预测概率
        tail_mask = bert_data['tail_mask'].to(self.device)
        attention_mask = bert_data['attention_mask'].to(self.device)
        labels = bert_data['labels'].to(self.device)
        result = self.bert_model(tail_mask, attention_mask, labels)
        tail_pos = np.array(bert_data['tail_pos'].squeeze()).tolist()
        # bert model
        bert_hidden = result.hidden_states[-1]
        bert_seq_hidden = torch.zeros((self.args.batch_size, 312)).to(self.device)
        for i in range(self.args.batch_size):
            bert_seq_hidden[i,:] = bert_hidden[i, tail_pos[i], :]
        bert_seq_hidden = bert_hidden[range(self.args.batch_size), tail_pos[i], :]
        bert_seq_hidden = self.bert_linear(bert_seq_hidden)

        # user embedding
        user_emb = self.user_embeddings(user_ids - self.args.num_locs) 
        user_emb = self.dropout(user_emb)
        user_emb_gate = user_emb.view(-1, self.args.hidden_size)           # [batch_size, hidden_size]

        bert_seq_hidden = torch.cat([bert_seq_hidden, user_emb_gate], dim=-1)
        # bert_seq_hidden = bert_seq_hidden
        return bert_seq_hidden

    def train_stage(self, data, bert_data):                                  # return loss
        targets = data[2]
        bert_seq_hidden, bert_loss = self.forward(data, bert_data)

        bert_seq_hidden = self.linear_final(bert_seq_hidden)
        bert_seq_hidden = self.dropout(bert_seq_hidden)                        # (batch_size, hidden_size)
        # 每一个POI的嵌入表示
        test_item_emb = self.loc_embeddings.weight[:self.args.num_locs]       # (num_poi, hidden_size)
        # calculate main loss, 寻找item_size个poi中与当前seq表征最相似的那个
        logits = torch.matmul(bert_seq_hidden, test_item_emb.transpose(0, 1))  # (batch_size, num_poi)
        main_loss = self.criterion(logits, targets)  

        joint_loss = 1 * main_loss + 0 * bert_loss
        return joint_loss                 

    def eval_stage(self, data, bert_batch):                                  # return logits / hidden
        bert_seq_hidden = self.forward_eval(data, bert_batch)
        bert_seq_hidden = self.linear_final(bert_seq_hidden)
        return bert_seq_hidden
    

def TransR_train(args, model, opt):
    Recmodel = model
    Recmodel.train()
    kgdataset = KG_data(args)
    kgloader = DataLoader(kgdataset, batch_size=256, drop_last=True)   
    trans_loss = 0.
    for data in tqdm(kgloader, total=len(kgloader), disable=True):
        heads = data[0].to(Recmodel.device)
        relations = data[1].to(Recmodel.device)
        pos_tails = data[2].to(Recmodel.device)
        neg_tails = data[3].to(Recmodel.device)
        kg_batch_loss = Recmodel.calc_kg_loss_transE(heads, relations, pos_tails, neg_tails)
        trans_loss += kg_batch_loss / len(kgloader)
        opt.zero_grad()
        kg_batch_loss.backward()
        opt.step()
    return trans_loss.cpu().item()
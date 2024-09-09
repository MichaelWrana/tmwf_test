#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_tmwf(traces, num_classes, traces_per_class, tabs):
    
    num_traces = 50000
    
    time = [None] * num_traces
    direction = [None] * num_traces
    label = [None] * num_traces
    truepos = [None] * num_traces
    
    final_traces = {'data':[None] * num_traces, 'time':[None] * num_traces, 'label': [None] * num_traces, }
    
    for i in tqdm(range(num_classes)):
        for j in range(traces_per_class):
            label = [-1] * tabs
            label[traces[i]['truepos'][j]] = i
        
            final_traces['data'][i*j] = traces[i]['direction'][j].astype(np.int8)
            final_traces['time'][i*j] = traces[i]['time'][j].astype(np.float16)
            final_traces['label'][i*j] = np.asarray(label, dtype=np.int8)
            
    return final_traces


def storage():
    train_loc = "../datasets/one_closed_new/3-tab/train/ieee_format/merged"
    test_loc = "../datasets/one_closed_new/3-tab/test/ieee_format/merged"
    with open(train_loc, 'rb') as f:
        train = pickle.load(f)

    with open(test_loc, 'rb') as f:
        test = pickle.load(f)
        
    train_ret = convert_to_tmwf(train, 50, 1000, 3)
    test_ret = convert_to_tmwf(test, 50, 100, 3)
    
    with open('train_ret', 'wb') as f:
        pickle.dump(train_ret, f)
        
    with open('test_ret', 'wb') as f:
        pickle.dump(test_ret, f)

class Transformer(nn.Module):

    def __init__(self, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()

        self.encoder = TransformerEncoder(TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_encoder_layers)
        self.decoder = TransformerDecoder(TransformerDecoderLayer(embed_dim, nhead, dim_feedforward, dropout),
                                          num_decoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        bs = src.shape[0]
        memory = self.encoder(src, pos_embed)
        query_embed = query_embed.repeat(bs, 1, 1)
        tgt = torch.zeros_like(query_embed)
        output = self.decoder(tgt, memory, pos_embed, query_embed)

        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos, query_pos):
        output = tgt

        for layer in self.layers:
            output = layer(output, memory, pos, query_pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, src, pos):
        src2 = self.self_attn(query=src + pos, key=src + pos, value=src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, pos, query_pos):
        tgt2 = self.self_attn(query=tgt + query_pos, key=tgt + query_pos, value=tgt)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=tgt + query_pos, key=memory + pos, value=memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class DFNet(nn.Module):
    def __init__(self, dropout):
        super(DFNet, self).__init__()

        # Block1
        filter_num = [0, 32, 64, 128, 256]
        kernel_size = [0, 8, 8, 8, 8]
        conv_stride_size = [0, 1, 1, 1, 1]
        pool_stride_size = [0, 4, 4, 4, 4]
        pool_size = [0, 8, 8, 8, 8]

        self.block1_conv1 = nn.Conv1d(in_channels=1, out_channels=filter_num[1],
                                      kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn1 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu1 = nn.ELU(alpha=1.0)
        self.block1_conv2 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[1], kernel_size=kernel_size[1],
                                      stride=conv_stride_size[1], padding=kernel_size[1] // 2)
        self.block1_bn2 = nn.BatchNorm1d(num_features=filter_num[1])
        self.block1_elu2 = nn.ELU(alpha=1.0)
        self.block1_pool = nn.MaxPool1d(kernel_size=pool_size[1], stride=pool_stride_size[1], padding=pool_size[1] // 2)
        self.block1_dropout = nn.Dropout(p=dropout)

        self.block2_conv1 = nn.Conv1d(in_channels=filter_num[1], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn1 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[2], kernel_size=kernel_size[2],
                                      stride=conv_stride_size[2], padding=kernel_size[2] // 2)
        self.block2_bn2 = nn.BatchNorm1d(num_features=filter_num[2])
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool1d(kernel_size=pool_size[2], stride=pool_stride_size[2], padding=pool_size[2] // 2)
        self.block2_dropout = nn.Dropout(p=dropout)

        self.block3_conv1 = nn.Conv1d(in_channels=filter_num[2], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn1 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[3], kernel_size=kernel_size[3],
                                      stride=conv_stride_size[3], padding=kernel_size[3] // 2)
        self.block3_bn2 = nn.BatchNorm1d(num_features=filter_num[3])
        self.block3_relu2 = nn.ReLU()
        self.block3_pool = nn.MaxPool1d(kernel_size=pool_size[3], stride=pool_stride_size[3], padding=pool_size[3] // 2)
        self.block3_dropout = nn.Dropout(p=dropout)

        self.block4_conv1 = nn.Conv1d(in_channels=filter_num[3], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn1 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = nn.Conv1d(in_channels=filter_num[4], out_channels=filter_num[4], kernel_size=kernel_size[4],
                                      stride=conv_stride_size[4], padding=kernel_size[4] // 2)
        self.block4_bn2 = nn.BatchNorm1d(num_features=filter_num[4])
        self.block4_relu2 = nn.ReLU()
        self.block4_pool = nn.MaxPool1d(kernel_size=pool_size[4], stride=pool_stride_size[4], padding=pool_size[4] // 2)
        self.block4_dropout = nn.Dropout(p=dropout)

    def forward(self, input):

        if len(input.shape) == 2:
            x = input.unsqueeze(1)
        else:
            x = input

        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_bn1(x)
        x = self.block1_elu1(x)
        x = self.block1_conv2(x)
        x = self.block1_bn2(x)
        x = self.block1_elu2(x)
        x = self.block1_pool(x)
        x = self.block1_dropout(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_bn1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_bn2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)
        x = self.block2_dropout(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_bn1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_bn2(x)
        x = self.block3_relu2(x)
        x = self.block3_pool(x)
        x = self.block3_dropout(x)

        # Block 4
        x = self.block4_conv1(x)
        x = self.block4_bn1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_bn2(x)
        x = self.block4_relu2(x)
        x = self.block4_pool(x)
        x = self.block4_dropout(x)
        return x.transpose(1, 2)
    


class TMWF_DFNet(nn.Module):

    def __init__(self, embed_dim, nhead, dim_feedforward, num_encoder_layers, num_decoder_layers, max_len, num_queries,
                 cls, dropout):
        super(TMWF_DFNet, self).__init__()
        print('TMWF_DFNet')
        self.cnn_layer = DFNet(dropout)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        self.trm = Transformer(embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.pos_embed = nn.Embedding(max_len, embed_dim).weight
        self.query_embed = nn.Embedding(num_queries, embed_dim).weight
        self.fc = nn.Linear(embed_dim, cls)

    def forward(self, input):
        x = self.cnn_layer(input)
        feat = self.proj(x)
        o = self.trm(feat, self.query_embed.unsqueeze(0), self.pos_embed.unsqueeze(0))
        logits = self.fc(o)

        return logits

class DataGenerator(object):
    def __init__(self, batch_size, dim, page, timestamp):
        # Initialization
        self.batch_size = batch_size
        self.dim = dim
        self.page = page
        self.timestamp = timestamp

    def generate(self, data, indices):
        max_len = self.dim * self.page
        total = len(indices)
        imax = int(len(indices) / self.batch_size)
        if total % self.batch_size != 0:
            imax = imax + 1

        while True:
            for i in range(imax):
                x = []
                y = []
                for j, k in enumerate(indices[i * self.batch_size:(i + 1) * self.batch_size]):

                    tlen = len(data['data'][k])
                    x_dire = data['data'][k]
                    if self.timestamp:
                        x_time = data['time'][k]
                        max_time = x_time[np.flatnonzero(x_time)[-1]]
                        for l, t in enumerate(x_time):
                            x_time[l] = t / max_time
                        x_time = np.array(x_time)
                        if tlen >= max_len:
                            x.append([x_dire.tolist()[:max_len], x_time.tolist()[:max_len]])
                        else:
                            x.append(
                                [x_dire.tolist() + [0] * (max_len - tlen), x_time.tolist() + [0] * (max_len - tlen)])
                    else:
                        if tlen >= max_len:
                            x.append(x_dire.tolist()[:max_len])
                        else:
                            x.append(x_dire.tolist() + [0] * (max_len - tlen))
                    y.append(data['label'][k])
                
                print(len(x))

                yield np.array(x), np.array(y)

def train_test(backbone, tab, page_dim, max_page, timestamp, train_ret, test_ret):
    np.random.seed(2023)
    torch.manual_seed(2023)

    train_total = len(train_ret['data'])
    indices = np.arange(train_total)
    np.random.shuffle(indices)
    train_gen = DataGenerator(batch_size, page_dim, max_page, timestamp).generate(train_ret, indices)

    test_total = len(test_ret['data'])
    indices = np.arange(test_total)
    np.random.shuffle(indices)
    test_gen = DataGenerator(batch_size, page_dim, max_page, timestamp).generate(test_ret, indices)
    
    if backbone == 'BAPM-CNN':
        model = TMWF_noDF(embed_dim=128, nhead=8, dim_feedforward=512, num_encoder_layers=2,
                          num_decoder_layers=2, max_len=page_dim * max_page // (8 * 8 * 8), num_queries=max_page,
                          cls=cls_num, dropout=0.1).cuda()
    elif backbone == 'DFNet':
        model = TMWF_DFNet(embed_dim=256, nhead=8, dim_feedforward=256 * 4, num_encoder_layers=2,
                           num_decoder_layers=2, max_len=121, num_queries=max_page, cls=cls_num,
                           dropout=0.1).cuda()
        
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criteron = torch.nn.CrossEntropyLoss()

    train_bc = train_total // batch_size
    if train_bc * batch_size != train_total:
        train_bc = train_bc + 1
    test_bc = test_total // batch_size
    if test_bc * batch_size != test_total:
        test_bc = test_bc + 1
        
    for epoch in range(50):
        model.train()
        count = 0
        for xb, yb in train_gen:
            log = model(torch.tensor(xb, dtype=torch.float).cuda())
            y_batch = torch.tensor(yb, dtype=torch.long).cuda()
            opt.zero_grad()
            loss = 0
            for ct in range(max_page):
                loss_ct = criteron(log[:, ct], y_batch[:, ct].cuda())
                loss = loss + loss_ct
            loss.backward()
            opt.step()
            count = count + 1
            if count == train_bc:
                break

        model.eval()
        probs = []
        one_hot = []
        count = 0
        for xb, yb in test_gen:
            log = model(torch.tensor(xb, dtype=torch.float).cuda())
            y_batch = torch.tensor(yb, dtype=torch.long).cuda()
            probs.append(log.data.cpu().numpy())
            one_hot.append(F.one_hot(y_batch, cls_num).data.cpu().numpy())
            count = count + 1
            if count == test_bc:
                break

        prob_matrix = np.concatenate(probs, axis=0)
        one_hot_matrix = np.concatenate(one_hot, axis=0)

        print('Epoch', epoch)

        print(prob_matrix)
        print(one_hot_matrix)

        #overall_basic_accuracy(prob_matrix, one_hot_matrix, cls_num)
        #overall_basic_precision(prob_matrix, one_hot_matrix, cls_num)
        #overall_basic_recall(prob_matrix, one_hot_matrix, cls_num)
        #overall_advanced_accuracy(prob_matrix, one_hot_matrix, cls_num)
        #overall_advanced_precision(prob_matrix, one_hot_matrix, cls_num)
        #overall_advanced_recall(prob_matrix, one_hot_matrix, cls_num)
        

def match_len(data):
    for i in range(len(data['data'])):
        if len(data['data'][i]) >= 5120:
            data['data'][i] = data['data'][i][:5120]
            data['time'][i] = data['time'][i][:5120]
        else:
            zeroes = np.asarray([0.0] * (5120 - len(data['data'][i])))
            data['data'][i] = np.concatenate((data['data'][i], zeroes))
            data['time'][i] = np.concatenate((data['time'][i], zeroes))

    return data

cls_num = 51
max_page = 3
page_len = 5120
batch_size = 80
lr = 0.0005


with open('train_ret', 'rb') as f:
    train_ret = pickle.load(f)
    
with open('test_ret', 'rb') as f:
    test_ret = pickle.load(f)


train_test(backbone='DFNet', tab=3, page_dim=page_len, max_page=max_page, timestamp=True, train_ret=train_ret, test_ret=test_ret)
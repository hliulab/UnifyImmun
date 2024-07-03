import copy
import math
from typing import Optional
import numpy as np
import torch
from torch import nn, Tensor
import pandas as pd
import torch.utils.data as Data

pep_max_len = 15
hla_max_len = 34
vocab = np.load('../data/data_dict.npy', allow_pickle=True).item()
vocab_size = len(vocab)
n_heads = 1
d_model = 64
d_ff = 512
d_k = d_v = 64
n_layers = 1
batch_size = 8192
epochs = 30
threshold = 0.5
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class Mymodel_HLA(nn.Module):
    def __init__(self,d_k =64, d_v = 64,d_model = 64,
                 n_heads = 1,n_layers = 1,d_ff = 512,hla_max_len = 34,pep_max_len = 15,tcr_max_len = 34):
        super(Mymodel_HLA, self).__init__()
        self.use_cuda = use_cuda
        self.encoder_H = Encoder().to(device)
        self.encoder_P = Encoder_padding().to(device)
        self.cross_1 = Cross_Attention().to(device)
        self.projection = nn.Sequential(
            nn.Linear(hla_max_len * d_model, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 2)
        ).to(device)

    def forward(self, pep_inputs, hla_inputs):
        hla_enc,hla_attn = self.encoder_H(hla_inputs)
        pep_enc,enc1_attn = self.encoder_P(pep_inputs)
        pep_hla, pep_hla_attn = self.cross_1(pep_enc,hla_enc)
        pep_hla_outputs = pep_hla.view(pep_hla.shape[0], -1)
        pep_hla_logits = self.projection(pep_hla_outputs)
        return pep_hla_logits.view(-1, pep_hla_logits.size(-1)), pep_hla_attn



def data_process_HLA(data):
    # print(data.columns)
    pep_inputs, hla_inputs, labels = [], [], []
    for pep, hla, label in zip(data.peptide, data.HLA, data.label):
        pep, hla = pep.ljust(hla_max_len, '-'), hla.ljust(hla_max_len, '-')
        pep_input = [[vocab[n] for n in pep]]  # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)
    return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)


class MyDataSet_HLA(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels):
        super(MyDataSet_HLA, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self):
        return self.pep_inputs.shape[0]

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]


def data_load_HLA(type_='train', fold=None, batch_size=batch_size):
    if type_ != 'train' and type_ != 'val':
        data = pd.read_csv('../data/data_HLA/{}_set.csv'.format(type_))
    elif type_ == 'train':
        data = pd.read_csv('../data/data_HLA/train_fold_{}.csv'.format(fold))
    elif type_ == 'val':
        data = pd.read_csv('../data/data_HLA/val_fold_{}.csv'.format(fold))
    pep_inputs, hla_inputs, labels = data_process_HLA(data)
    loader = Data.DataLoader(MyDataSet_HLA(pep_inputs, hla_inputs, labels), batch_size, shuffle=False, num_workers=0,drop_last=True)
    return loader

def transfer(y_prob, threshold=0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=34):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding_padding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding_padding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pad = torch.zeros(34,d_model)
        pad[:pe.shape[0], :] = pe
        pe = pad.unsqueeze(0).transpose(0, 1).to(device)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x.to(device) + self.pe[:x.size(0), :].to(device)
        return self.dropout(x)

def get_attn_pad_mask(seq_q,seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # print(pad_attn_mask.size())
    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # batch_size, n_heads, len_q, len_k
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # batch_size, n_heads, len_q, d_v
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.use_cuda = use_cuda
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,n_heads * d_v)
        output = self.fc(context)  # batch_size, len_q, d_model
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.use_cuda = use_cuda
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        output = nn.Dropout(0.1)(output)
        return nn.LayerNorm(d_model).to(device)(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(0.1)
    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,enc_self_attn_mask)
        enc_outputs1 = enc_inputs + self.dropout(enc_outputs)
        enc_outputs1 = nn.LayerNorm(d_model).to(device)(enc_outputs1)
        enc_outputs = self.pos_ffn(enc_outputs1)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        # print(enc_inputs.size())
        # print(enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: batch_size, src_len, d_model, enc_self_attn: batch_size, n_heads, src_len, src_len
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Encoder_padding(nn.Module):
    def __init__(self):
        super(Encoder_padding, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb_padding = PositionalEncoding_padding(d_model,max_len=pep_max_len)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        enc_outputs = self.src_emb(enc_inputs)  # batch_size, src_len, d_model
        enc_pad = torch.zeros(batch_size,hla_max_len,d_model)
        # print(enc_outputs.size())
        enc_pad[:, :enc_outputs.shape[1], :] = enc_outputs
        enc_outputs = enc_pad
        enc_outputs = self.pos_emb_padding(enc_outputs.transpose(0, 1)).transpose(0, 1)  # batch_size, src_len, d_model
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # batch_size, src_len, src_len
        # print(enc_inputs.size())
        # print(enc_self_attn_mask.size())
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: batch_size, src_len, d_model, enc_self_attn: batch_size, n_heads, src_len, src_len
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.dropout = nn.Dropout(0.1)
    def forward(self, pep_inputs, HLA_inputs, dec_self_attn_mask):
        # dec_outputs: batch_size, tgt_len, d_model, dec_self_attn: batch_size, n_heads, tgt_len, tgt_len
        # print(pep_inputs.size())
        # print(HLA_inputs.size())
        # print(dec_self_attn_mask.size())
        dec_outputs, dec_self_attn = self.dec_self_attn(pep_inputs, HLA_inputs, HLA_inputs, dec_self_attn_mask)
        dec_outputs = self.dropout(dec_outputs)
        dec_outputs = self.pos_ffn(dec_outputs)  # batch_size, tgt_len, d_model
        return dec_outputs, dec_self_attn



class Cross_Attention(nn.Module):
    def __init__(self):
        super(Cross_Attention, self).__init__()
        self.use_cuda = use_cuda
        self.pos_emb = PositionalEncoding(d_model)
        self.pos_peptide = PositionalEncoding_padding(d_model,max_len=15)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])
        self.tgt_len = hla_max_len

    def forward(self, pep_inputs,HLA_inputs):
        pep_outputs = pep_inputs.to(device)
        HLA_outputs = HLA_inputs.to(device)
        dec_self_attn_pad_mask = torch.LongTensor(np.zeros((pep_inputs.shape[0], hla_max_len, hla_max_len))).bool().to(device)
        dec_self_attns = []
        for layer in self.layers:
            # dec_outputs: batch_size, tgt_len, d_model
            # print(dec_self_attn_pad_mask.size())
            dec_outputs, dec_self_attn = layer(pep_outputs, HLA_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns


def with_pos_embed(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

def _get_activation_fn(activation):
    if activation == "relu":
        return nn.functional.relu
    if activation == "gelu":
        return nn.functional.gelu
    if activation == "glu":
        return nn.functional.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

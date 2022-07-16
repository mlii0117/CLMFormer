import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import infEncoder, infEncoderLayer, ConvLayer,transEncoder, transEncoderLayer
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
from models.relationalmemory import RelationalMemory
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True,
                device=torch.device('cuda:0'),rm_num_slots=3,rm_d_model=512,rm_num_heads=8,path='./'):
        super(Informer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.dropout=dropout
        self.ff = PositionwiseFeedForward(d_model, d_ff, self.dropout)
        c = copy.deepcopy

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, self.dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, self.dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention

        #memory
        self.rm_num_slots=rm_num_slots
        self.rm_d_model=rm_d_model
        self.rm_num_heads=rm_num_heads
        self.rm=RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads,path=path)
        
        # Encoder
        self.encoder = infEncoder(
            [
                infEncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=self.dropout, output_attention=output_attention),
                                d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=self.dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder =Decoder(
                DecoderLayer(d_model, AttentionLayer(Attn(True, factor, attention_dropout=self.dropout, output_attention=False), d_model, n_heads),
                             AttentionLayer(FullAttention(False, factor, attention_dropout=self.dropout, output_attention=False), d_model, n_heads),
                             c(self.ff), self.dropout, self.rm_num_slots, self.rm_d_model),d_layers)


        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,memory,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        if memory == None:
            memory = self.rm.init_memory(enc_out.size(0)).to(enc_out)

        #print('mem2',memory.size())
        memory = self.rm(self.dec_embedding(x_dec,x_mark_dec), memory)
        #print('x_dec', x_dec.size())
        dec_out = self.decoder(self.dec_embedding(x_dec,x_mark_dec), enc_out, dec_self_mask, dec_enc_mask,memory)
        

        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Transformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True,
                 device=torch.device('cuda:0'), rm_num_slots=3, rm_d_model=512, rm_num_heads=8, path='./'):
        super(Transformer, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self.dropout = dropout
        self.ff = PositionwiseFeedForward(d_model, d_ff, self.dropout)
        c = copy.deepcopy
        # Encoding
        # print('dropout', dropout)

        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, self.dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, self.dropout)
        # Attention
        Attn = ProbAttention if attn == 'prob' else FullAttention

        # memory
        self.rm_num_slots = rm_num_slots
        self.rm_d_model = rm_d_model
        self.rm_num_heads = rm_num_heads
        self.rm = RelationalMemory(num_slots=self.rm_num_slots, d_model=self.rm_d_model, num_heads=self.rm_num_heads,
                                   path=path)

        # Encoder
        self.encoder = transEncoder(

            transEncoderLayer(
                AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                               d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout
            ), e_layers)
        # Decoder
        self.decoder = Decoder(
            DecoderLayer(d_model,
                         AttentionLayer(Attn(True, factor, attention_dropout=self.dropout, output_attention=False),
                                        d_model, n_heads),
                         AttentionLayer(
                             FullAttention(False, factor, attention_dropout=self.dropout, output_attention=False),
                             d_model, n_heads),
                         c(self.ff), self.dropout, self.rm_num_slots, self.rm_d_model), d_layers)

        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, memory,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(enc_out, enc_self_mask)
        # print('mem1',enc_out.size())

        if memory == None:
            memory = self.rm.init_memory(enc_out.size(0)).to(enc_out)

        # print('mem2',memory.size())
        memory = self.rm(self.dec_embedding(x_dec, x_mark_dec), memory)
        # print('x_dec', x_dec.size())
        dec_out = self.decoder(self.dec_embedding(x_dec, x_mark_dec), enc_out, dec_self_mask, dec_enc_mask, memory)

        dec_out = self.projection(dec_out)

        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


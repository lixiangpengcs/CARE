import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class RelationEmbedding(nn.Module):
    def __init__(self, hparams):
        super(RelationEmbedding, self).__init__()
        print("Building RelationEmbedding")
        self.reg2grid = SelfAtt(d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, attention_mask=None)
        self.grid2reg = SelfAtt(d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, attention_mask=None)
        self.hparams = hparams
        img_feature_size = 2048
        self.linear_grid = nn.Linear(img_feature_size, hparams.hidden_size)
        self.linear_grid_dropout = nn.Dropout(0.10)
        self.grid_layernorm = nn.LayerNorm(hparams.hidden_size)
        self.grid_embedding = nn.Linear(4, 512)
        self.linear_region = nn.Linear(img_feature_size, hparams.hidden_size)
        self.box_embedding = nn.Linear(4, 512)
        self.linear_region_dropout = nn.Dropout(0.10)
        self.region_layernorm = nn.LayerNorm(hparams.hidden_size)

    def forward(self, batch):
        reg_feat = batch['img_feat'] # bs x N x 2048
        reg_bbox = batch['region_bbox'] # bs x N x 4
        grid_feat = batch['grid_feat'] # bs x 49 x 2048
        grid_bbox = batch['grid_bbox'] # bs x 49 x 4

        batch_size = reg_feat.size(0)

        reg_feat_proj = self.region_layernorm(self.linear_region(reg_feat) + self.box_embedding(reg_bbox)) # bs x N x 512
        grid_feat_proj = self.grid_layernorm(self.linear_grid(grid_feat) + self.grid_embedding(grid_bbox)) # bs x 49 x 512

        context_feat_proj = torch.cat([reg_feat_proj, grid_feat_proj], dim=-2)

        reg_mask = (0 != reg_feat.abs().sum(-1)).unsqueeze(1) # bs x 1 x 100
        grid_mask = (0 != grid_feat.abs().sum(-1)).unsqueeze(1) # bs x 1 x 49
        # aligns = batch['aligns'].unsqueeze(1).bool()  # bs x 100 x 49
        # grid_out = self.reg2grid(grid_feat_proj, reg_feat_proj, reg_feat_proj, aligns.permute(0,1,3,2)) + grid_feat_proj  #  bs x 49 x obj_num
        # reg_out = self.grid2reg(reg_feat_proj, grid_feat_proj, grid_feat_proj, aligns) + reg_feat_proj

        aligns = batch['aligns']
        grid_ones = torch.eye(grid_feat_proj.size(1)).to(reg_feat.device) # 49 x 49
        grid_ones_extend = grid_ones.unsqueeze(0).repeat(batch_size, 1, 1) # 1 x 49 x 49
        grid_aligns = torch.cat([aligns, grid_ones_extend], dim=-2).unsqueeze(1).bool() # bs x 1 x 149 x 49

        region_ones = torch.eye(reg_feat_proj.size(1)).to(reg_feat.device)  # 100 x 100
        region_ones_extend = region_ones.unsqueeze(0).repeat(batch_size, 1, 1)  # bs x 100 x 00
        region_aligns = torch.cat([aligns, region_ones_extend], dim=-1).unsqueeze(1).bool() # bs x 1 x 100 x 149

        grid_out = self.reg2grid(grid_feat_proj, context_feat_proj, context_feat_proj, grid_aligns.permute(0,1,3,2)) + grid_feat_proj  #  bs x 49 x 100
        reg_out = self.grid2reg(reg_feat_proj, context_feat_proj, context_feat_proj, region_aligns) + reg_feat_proj

        out_feat = torch.cat([reg_out, grid_out], dim=1)
        out_mask = torch.cat([reg_mask, grid_mask], dim=-1) # bs x 1 x (obj_num+49)

        return out_feat, out_mask

class QuesCondRelationEmbedding(nn.Module):
    def __init__(self, hparams):
        super(QuesCondRelationEmbedding, self).__init__()
        self.ques_linear = nn.Linear(hparams.lstm_hidden_size*2, hparams.hidden_size)
        self.reg2grid = SelfAtt(d_model=hparams.hidden_size, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, attention_mask=None)
        self.grid2reg = SelfAtt(d_model=hparams.hidden_size, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, attention_mask=None)

        self.hparams = hparams
        self.linear_grid = nn.Linear(hparams.img_feature_size, hparams.hidden_size)
        self.linear_grid_dropout = nn.Dropout(0.10)
        self.grid_layernorm = nn.LayerNorm(hparams.hidden_size)
        self.grid_embedding = nn.Linear(4, hparams.hidden_size)
        self.linear_region = nn.Linear(hparams.img_feature_size, hparams.hidden_size)
        self.box_embedding = nn.Linear(4, hparams.hidden_size)
        self.linear_region_dropout = nn.Dropout(0.10)
        self.region_layernorm = nn.LayerNorm(hparams.hidden_size)

    def forward(self, batch, ques_repr):
        # ques_repr: bs x 10 x hidden_size
        reg_feat = batch['img_feat']  # bs x N x 2048
        reg_bbox = batch['region_bbox']  # bs x N x 4
        grid_feat = batch['grid_feat']  # bs x 49 x 2048
        grid_bbox = batch['grid_bbox']  # bs x 49 x 4

        reg_feat_proj = self.region_layernorm(
            self.linear_region(reg_feat) + self.box_embedding(reg_bbox))  # bs x N x 512
        grid_feat_proj = self.grid_layernorm(
            self.linear_grid(grid_feat) + self.grid_embedding(grid_bbox))  # bs x 49 x 512

        bs, reg_num = reg_feat.shape[:2]
        grid_num = grid_feat.shape[1]
        sent_len = ques_repr.shape[1]

        ques_repr_pro = self.ques_linear(ques_repr)
        reg_mask = (0 != reg_feat.abs().sum(-1)).unsqueeze(1)  # bs x 1 x obj_num
        grid_mask = (0 != grid_feat.abs().sum(-1)).unsqueeze(1)  # bs x 1 x 49

        reg_feat_unsq = reg_feat_proj.unsqueeze(2)   # bs x num x 1 x 512
        grid_feat_unsq = grid_feat_proj.unsqueeze(2) # bs x 49 x 1 x 512
        ques_repr_unsq = ques_repr_pro.unsqueeze(1) # bs x 1 x 10 x hidden_size

        # reg_feat_con = torch.cat([reg_feat_unsq, ques_repr_unsq], dim=-1).view(bs, reg_num*sent_len, -1)  # bs x num x 10 x (hidden_size+dim) => bs x (numx10) x (hidden_size+dim)
        # grid_feat_con = torch.cat([grid_feat_unsq, ques_repr_unsq], dim=-1).view(bs, grid_num*sent_len, -1) # bs x 49 x 10 x (hidden_size+dim) => bs x (49x10) x (hidden_size+dim)
        reg_feat_con = (reg_feat_unsq+ ques_repr_unsq).view(bs, reg_num * sent_len, -1)  # bs x num x 10 x (hidden_size+dim) => bs x (numx10) x (hidden_size+dim)
        grid_feat_con = (grid_feat_unsq+ ques_repr_unsq).view(bs, grid_num * sent_len, -1)  # bs x 49 x 10 x (hidden_size+dim) => bs x (49x10) x (hidden_size+dim)

        aligns = batch['aligns'].unsqueeze(2).unsqueeze(4)  # bs x obj_num x 1 x 49 x 1
        extend_aligns = aligns.expand(-1, -1, sent_len, -1, sent_len)
        extend_aligns = extend_aligns.reshape(bs, sent_len*reg_num, sent_len*grid_num)
        extend_aligns = extend_aligns.unsqueeze(1).bool()
        grid_out = self.reg2grid(grid_feat_con, reg_feat_con, reg_feat_con,
                                 extend_aligns.permute(0, 1, 3, 2)) + grid_feat_con  # bs x 49 x obj_num
        reg_out = self.grid2reg(reg_feat_con, grid_feat_con, grid_feat_con, extend_aligns) + reg_feat_con

        out_feat = torch.cat([reg_out, grid_out], dim=1)
        out_feat_view = out_feat.view(bs, (reg_num+grid_num), sent_len, -1)
        out_mask = torch.cat([reg_mask, grid_mask], dim=-1)  # bs x 1 x (obj_num+49)

        return out_feat_view, out_mask


class SelfAtt(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1, attention_mask=None):
        super(SelfAtt, self).__init__()
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask):
        atten_feat = self.mhatt(queries, keys, values, attention_mask)
        att = self.layernorm(queries + atten_feat)
        att_d = self.dropout(att)
        out = self.ffn(att_d)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout):
        super(MultiHeadAttention, self).__init__()
        self.sdpa = ScaleDotProductAttention(d_model, d_k, d_v, h, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, attention_mask):
        out = self.sdpa(queries, keys, values, attention_mask)
        out = self.dropout(out)
        out = self.layernorm(out)
        return out

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
        out = self.dropout(out)
        out = self.layer_norm(input + out)
        return out

class ScaleDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.fc_q = nn.Linear(d_model, h*d_k)
        self.fc_k = nn.Linear(d_model, h*d_k)
        self.fc_v = nn.Linear(d_model, h*d_v)
        self.fc_o = nn.Linear(h*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask):
        bs, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(bs, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(bs, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(bs, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)

        if attention_mask is not None:
            pad_attention_mask = ~attention_mask
            att = att.masked_fill(pad_attention_mask, -10000.0)

        att = F.softmax(att, -1)
        att = self.dropout(att)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(bs, nq, self.h * self.d_v)
        return out

if __name__ == '__main__':
    a = torch.rand([8, 13, 16])
    b = torch.rand([8, 14, 16])

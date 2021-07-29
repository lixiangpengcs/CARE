import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig, BertPreTrainedModel, BertEmbeddings
import math
import numpy as np

class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, txt_inds, txt_mask):
        encoder_inputs = self.embeddings(txt_inds)
        attention_mask = txt_mask

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output

class HistoryAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(HistoryAttention, self).__init__()
        self.linear_hist = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.15)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, hist_word_encoded, accu_h_not_pad):
        # hist_word_encoded: bs x 10 x 40 x 1024, hist_mask: bs x 10 x 40 x 1
        if len(accu_h_not_pad.size()) == 2:
            accu_h_not_pad = accu_h_not_pad.unsqueeze(-1)
        masked_values = (accu_h_not_pad.float() -1.0) * 10000.0
        atten_values = self.dropout(self.linear_hist(hist_word_encoded)) + masked_values
        atten_weights = self.softmax(atten_values)
        attended_hist = (atten_weights * hist_word_encoded).sum(1)
        return attended_hist

class ContextGate(nn.Module):
    def __init__(self, hidden_dim):
        super(ContextGate, self).__init__()
        self.linear_trans = nn.Linear(hidden_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, ctx):
        proj_ctx = self.sigmoid(self.linear_trans(ctx))
        return proj_ctx * ctx

class FusionForgetGate(nn.Module):
    def __init__(self, hidden_dim):
        super(FusionForgetGate, self).__init__()
        self.linear_ctx = nn.Linear(hidden_dim*2, hidden_dim)
        self.linear_condition = nn.Linear(hidden_dim, hidden_dim)
        self.cond_aggr = HistoryAttention(hidden_dim)
        self.sigmoid = nn.Sigmoid()

        self.linear_memory = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, condition, ctx):
        # condition: bs x q_len x dim    ctx: bs x len x dim
        condition_aggr = self.cond_aggr(condition).unsqueeze(1) # bs x 1 x dim
        concat_ctx = torch.cat([ctx, condition_aggr], dim=-1) # bs x q_len x (2*dim)
        proj_ctx = self.linear_ctx(concat_ctx) # bs x q_len x dim
        ffg_values = self.sigmoid(proj_ctx)

        ctx_memory = self.linear_memory(ctx)
        return ffg_values * ctx_memory
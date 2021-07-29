import torch
from torch import nn
from torch.nn.functional import normalize
from transformers.models.bert.modeling_bert import BertEncoder, BertConfig, BertPreTrainedModel, BertEmbeddings
from visdial.encoders.transformermodel.modules import HistoryAttention, ContextGate, FusionForgetGate
from visdial.utils import DynamicRNN
BertLayerNorm = torch.nn.LayerNorm
import torch.nn.functional as F

from visdial.utils.relationembedding import RelationEmbedding, QuesCondRelationEmbedding

class TransformerModelEncoder(nn.Module):
    def __init__(self, hparams, vocabulary):
        super(TransformerModelEncoder, self).__init__()
        BERT_HIDDEN_SIZE = 768
        self.hparams = hparams
        self.hidden_size = hparams.hidden_size

        self.word_embed = nn.Embedding(
            len(vocabulary),
            hparams.word_embedding_size,
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.ques_rnn = nn.LSTM(
            hparams.word_embedding_size,
            hparams.lstm_hidden_size,
            hparams.lstm_num_layers,
            batch_first=True,
            dropout=hparams.dropout,
            bidirectional=True
        )
        self.hist_rnn = nn.LSTM(
            hparams.word_embedding_size,
            hparams.lstm_hidden_size,
            hparams.lstm_num_layers,
            batch_first=True,
            dropout=hparams.dropout,
            bidirectional=True
        )

        self.correlation = RelationEmbedding(hparams)
        self.linear_img = nn.Linear(hparams.hidden_size, BERT_HIDDEN_SIZE)
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        self.linear_ques = nn.Linear(self.hidden_size*2, BERT_HIDDEN_SIZE)
        self.ques_drop = nn.Dropout(0.15)
        self.linear_hist = nn.Linear(self.hidden_size*2, BERT_HIDDEN_SIZE)
        self.hist_drop = nn.Dropout(0.15)

        self.linear_obj_feat = nn.Linear(self.hparams.img_feature_size, self.hidden_size)
        self.linear_obj_box = nn.Linear(4, self.hidden_size)
        self.obj_layernorm = BertLayerNorm(self.hidden_size)
        self.obj_box_layernorm = BertLayerNorm(self.hidden_size)

        # history
        self.histatten = HistoryAttention(hparams.hidden_size*2)
        self.QIHBert_config = BertConfig(num_hidden_layers=hparams.qih_bert_layers)
        self.QIHBert = BertEncoder(self.QIHBert_config)

        # fusion
        self.fusion = nn.Linear(BERT_HIDDEN_SIZE, 512)
        self.fusion_gate = nn.Linear(512, 512)

        # self selection
        self.qih_selection = HistoryAttention(BERT_HIDDEN_SIZE)
        self.qih_gate = ContextGate(BERT_HIDDEN_SIZE)

    def forward(self, batch):
        img, img_mask = self.init_img(batch)
        img_mask = img_mask.squeeze(1)
        img = self.linear_img(img)
        _, num_p, img_feat_size = img.size()

        ques_word_embed, ques_word_encoded, ques_encoded, ques_not_pad, ques_pad = self.init_q_embed(batch)
        bs, num_r, bilstm = ques_encoded.size()  # 320 x 10 x 1024
        ques_word_encoded = ques_word_encoded.view(bs, num_r, -1, bilstm)
        ques_not_pad = ques_not_pad.view(bs, num_r, -1)

        proj_ques_word_encoded = self.ques_drop(self.linear_ques(ques_word_encoded))

        # history preprocess
        hist_word_embed, hist_word_encoded, hist_encoded, hist_not_pad, hist_pad = self.init_h_embed(batch)
        # hist_word_encoded = hist_word_encoded.view(bs, num_r, -1, bilstm)
        hist_encoded = self.histatten(hist_word_encoded, hist_not_pad)  # (bsx10) x 1 x bilstm

        hist_encoded = hist_encoded.view(bs, num_r, -1)  # bs x 10 x 1024
        proj_hist_encoded = self.hist_drop(self.linear_hist(hist_encoded))
        qih_outs = []
        for c_r in range(num_r):  # current round
            hist_sent_encoded = proj_hist_encoded[:, 0:(c_r + 1), :]  # bs x sl_q x dim
            current_hist_mask = torch.ones(bs, c_r + 1).to(hist_sent_encoded.device)
            curr_q_encoded = proj_ques_word_encoded[:, c_r:(c_r + 1), :, :].squeeze(
                1)  # bs x 1 x q_len x dim => bs x q_len x dim
            current_q_mask = ques_not_pad[:, c_r:(c_r + 1), :].squeeze(1)

            qih_inputs = torch.cat([curr_q_encoded, img, hist_sent_encoded],
                                  dim=-2)  # [bs x q_len x dim, bs x obj_num x dim] => bs x (q_len+obj_num) x dim
            qih_mask = torch.cat([current_q_mask, img_mask, current_hist_mask], dim=-1).float()
            extend_qih_mask = qih_mask.unsqueeze(1).unsqueeze(2)
            extend_qih_mask = (1.0 - extend_qih_mask) * -10000.0
            qih_out = self.QIHBert(qih_inputs, extend_qih_mask)
            qih_outs.append(self.qih_gate(qih_out[0][:, 0, :]).unsqueeze(1))
        qih_summary = torch.cat(qih_outs, dim=1)
        qih_predict = self.fusion_gate(self.fusion(qih_summary))
        return qih_predict

    def init_img(self, batch):
        if self.hparams.grid_feat_trigger and self.hparams.grid_region_trigger:
            img, mask = self.correlation(batch)
            # reg_img = batch['img_feat']
            # grid_img = batch['grid_feat']
            # img = torch.cat([reg_img, grid_img], dim=1)
        elif self.hparams.grid_region_trigger:
            reg_img = batch['grid_feat']
            reg_boxes = F.normalize(batch['bbox'], dim=-1)
            img = self.linear_region_dropout(
                self.linear_region(reg_img) + self.linear_box_dropout(self.linear_reg_box(reg_boxes)))
        else:
            img = batch['img_feat']
        """image feature normarlization"""
        if self.hparams.img_norm:
            img = normalize(img, dim=1, p=2)
        # mask = (0 != img.abs().sum(-1))
        return img, mask

    def init_q_embed(self, batch):
        ques = batch['ques']
        bs, nr, sl_q = ques.size()
        lstm = self.hparams.lstm_hidden_size
        """bs_q, nr_q, sl_q -> bs*nr, sl_q, 1"""
        ques_not_pad = (ques != 0).bool()
        ques_not_pad = ques_not_pad.view(-1, sl_q).unsqueeze(-1)
        ques_pad = (ques == 0).bool()
        ques_pad = ques_pad.view(-1, sl_q).unsqueeze(1)

        ques = ques.view(-1, sl_q)
        ques_word_embed = self.word_embed(ques)
        ques_word_encoded, _ = self.ques_rnn(ques_word_embed, batch['ques_len'])

        loc = batch['ques_len'].view(-1).cpu().numpy() - 1

        # sentence-level encoded
        ques_encoded_forawrd = ques_word_encoded[range(bs * nr), loc, :lstm]
        ques_encoded_backward = ques_word_encoded[:, 0, lstm:]
        ques_encoded = torch.cat((ques_encoded_forawrd, ques_encoded_backward), dim=-1)
        ques_encoded = ques_encoded.view(bs, nr, -1)

        return ques_word_embed, ques_word_encoded, ques_encoded, ques_not_pad, ques_pad

    def init_h_embed(self, batch):
        hist = batch['hist']
        bs, nr, sl_h = hist.size()
        lstm = self.hparams.lstm_hidden_size
        """bs_q, nr_q, sl_q -> bs*nr, sl_q, 1"""
        hist_not_pad = (hist != 0).bool()
        hist_not_pad = hist_not_pad.view(-1, sl_h).unsqueeze(-1)
        hist_pad = (hist == 0).bool()
        hist_pad = hist_pad.view(-1, sl_h).unsqueeze(1)

        hist = hist.view(-1, sl_h)  # bs*nr,sl_q
        hist_word_embed = self.word_embed(hist)  # bs*nr,sl_q, emb_s
        hist_word_encoded, _ = self.hist_rnn(hist_word_embed, batch['hist_len'])

        loc = batch['hist_len'].view(-1).cpu().numpy() - 1

        # sentence-level encoded
        hist_encoded_forawrd = hist_word_encoded[range(bs * nr), loc, :lstm]
        hist_encoded_backward = hist_word_encoded[:, 0, lstm:]
        hist_encoded = torch.cat((hist_encoded_forawrd, hist_encoded_backward), dim=-1)
        hist_encoded = hist_encoded.view(bs, nr, -1)

        return hist_word_embed, hist_word_encoded, hist_encoded, hist_not_pad, hist_pad
import torch
from torch import nn
from torch.nn import functional as F


class GatedTrans(nn.Module):
	"""
		original code is from https://github.com/yuleiniu/rva (CVPR, 2019)
		They used tanh and sigmoid, but we used tanh and LeakyReLU for non-linear transformation function
	"""
	def __init__(self, in_dim, out_dim):
		super(GatedTrans, self).__init__()
		self.embed_y = nn.Sequential(
			nn.Linear(
				in_dim,
				out_dim
			),
			nn.Tanh()
		)
		self.embed_g = nn.Sequential(
			nn.Linear(
				in_dim,
				out_dim
			),
			nn.LeakyReLU()
		)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.kaiming_uniform_(m.weight.data)
				if m.bias is not None:
					nn.init.constant_(m.bias.data, 0)

	def forward(self, x_in):
		x_y = self.embed_y(x_in)
		x_g = self.embed_g(x_in)
		x_out = x_y * x_g

		return x_out


class Q_ATT(nn.Module):
    """Self attention module of questions."""

    def __init__(self, hparams):
        super(Q_ATT, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques_word, ques_word_encoded, ques_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim)
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2)
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max)
        # output: img_att (batch_size, num_rounds, embed_dim)
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = self.embed(ques_word_encoded)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)

        att = self.att(ques_norm).squeeze(-1)  # shape: (batch_size, num_rounds, quen_len_max)
        # ignore <pad> word
        att = self.softmax(att)
        att = att * ques_not_pad  # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True)  # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2)  # shape: (batch_size, num_rounds, rnn_dim)

        return feat, att


class H_ATT(nn.Module):
    """question-based history attention"""

    def __init__(self, hparams):
        super(H_ATT, self).__init__()

        self.H_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, hist, ques):
        # hist shape: (batch_size, num_rounds, rnn_dim)
        # ques shape: (batch_size, num_rounds, rnn_dim)
        # output: hist_att (batch_size, num_rounds, embed_dim)
        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        hist_embed = self.H_embed(hist)  # shape: (batch_size, num_rounds, embed_dim)
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, embed_dim)

        ques_embed = self.Q_embed(ques)  # shape: (batch_size, num_rounds, embed_dim)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, embed_dim)

        att_embed = F.normalize(hist_embed * ques_embed, p=2, dim=-1)  # (batch_size, num_rounds, num_rounds, embed_dim)
        att_embed = self.att(att_embed).squeeze(-1)
        att = self.softmax(att_embed)  # shape: (batch_size, num_rounds, num_rounds)
        att_not_pad = torch.tril(
            torch.ones(size=[num_rounds, num_rounds], requires_grad=False))  # shape: (num_rounds, num_rounds)
        att_not_pad = att_not_pad.cuda()
        att_masked = att * att_not_pad  # shape: (batch_size, num_rounds, num_rounds)
        att_masked = att_masked / torch.sum(att_masked, dim=-1,
                                            keepdim=True)  # shape: (batch_size, num_rounds, num_rounds)
        feat = torch.sum(att_masked.unsqueeze(-1) * hist.unsqueeze(1),
                         dim=-2)  # shape: (batch_size, num_rounds, rnn_dim)

        return feat


class V_Filter(nn.Module):
    """docstring for V_Filter"""

    def __init__(self, hparams):
        super(V_Filter, self).__init__()

        self.filter = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.word_embedding_size,
                hparams.img_feature_size
            ),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, img, ques):
        # img shape: (batch_size, num_rounds, i_dim)
        # ques shape: (batch_size, num_rounds, q_dim)
        # output: img_att (batch_size, num_rounds, embed_dim)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = self.filter(ques)  # shape: (batch_size, num_rounds, embed_dim)

        # gated
        img_fused = img * ques_embed

        return img_fused


class ATT_MODULE(nn.Module):
    """docstring for ATT_MODULE"""

    def __init__(self, hparams):
        super(ATT_MODULE, self).__init__()

        self.V_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.img_feature_size,
                hparams.lstm_hidden_size
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.word_embedding_size,
                hparams.lstm_hidden_size
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                1
            )
        )

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, img, ques):
        # input
        # img - shape: (batch_size, num_proposals, img_feature_size)
        # ques - shape: (batch_size, num_rounds, word_embedding_size)
        # output
        # att - shape: (batch_size, num_rounds, num_proposals)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        num_proposals = img.size(1)

        img_embed = img.view(-1, img.size(-1))  # shape: (batch_size * num_proposals, img_feature_size)
        img_embed = self.V_embed(img_embed)  # shape: (batch_size, num_proposals, lstm_hidden_size)
        img_embed = img_embed.view(batch_size, num_proposals,
                                   img_embed.size(-1))  # shape: (batch_size, num_proposals, lstm_hidden_size)
        img_embed = img_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                  1)  # shape: (batch_size, num_rounds, num_proposals, lstm_hidden_size)

        ques_embed = ques.view(-1, ques.size(-1))  # shape: (batch_size * num_rounds, word_embedding_size)
        ques_embed = self.Q_embed(ques_embed)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.view(batch_size, num_rounds,
                                     ques_embed.size(-1))  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_proposals,
                                                    1)  # shape: (batch_size, num_rounds, num_proposals, lstm_hidden_size)

        att_embed = F.normalize(img_embed * ques_embed, p=2,
                                dim=-1)  # (batch_size, num_rounds, num_proposals, lstm_hidden_size)
        att_embed = self.att(att_embed).squeeze(-1)  # (batch_size, num_rounds, num_proposals)
        att = self.softmax(att_embed)  # shape: (batch_size, num_rounds, num_proposals)

        return att


class PAIR_MODULE(nn.Module):
    """docstring for PAIR_MODULE"""

    def __init__(self, hparams):
        super(PAIR_MODULE, self).__init__()

        self.H_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size),
        )
        self.MLP = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size * 2,
                hparams.lstm_hidden_size),
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                1
            )
        )
        self.att = nn.Linear(2, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, hist, ques):
        # input
        # ques shape: (batch_size, num_rounds, lstm_hidden_size*2)
        # hist shape: (batch_size, num_rounds, lstm_hidden_size*2)
        # output
        # hist_gs_set - shape: (batch_size, num_rounds, num_rounds)

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        hist_embed = self.H_embed(hist)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, lstm_hidden_size)

        ques_embed = self.Q_embed(ques)  # shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds,
                                                    1)  # shape: (batch_size, num_rounds, num_rounds, lstm_hidden_size)

        att_embed = torch.cat((hist_embed, ques_embed), dim=-1)
        score = self.MLP(att_embed)

        delta_t = torch.tril(torch.ones(size=[num_rounds, num_rounds], requires_grad=False)).cumsum(
            dim=0)  # (num_rounds, num_rounds)
        delta_t = delta_t.view(1, num_rounds, num_rounds, 1).repeat(batch_size, 1, 1,
                                                                    1)  # (batch_size, num_rounds, num_rounds, 1)
        delta_t = delta_t.cuda()
        att_embed = torch.cat((score, delta_t), dim=-1)  # (batch_size, num_rounds, num_rounds, lstm_hidden_size*2)

        hist_logits = self.att(att_embed).squeeze(-1)  # (batch_size, num_rounds, num_rounds)

        # PAIR
        hist_gs_set = torch.zeros_like(hist_logits)
        for i in range(num_rounds):
            # one-hot
            logits = hist_logits[:, i, :(i + 1)]
            if self.training:
                hist_gs = F.gumbel_softmax(logits, hard=True)  # shape: (batch_size, i+1)
            else:
                _, max_value_indexes = logits.detach().max(1, keepdim=True)
                hist_gs = logits.detach().clone().zero_().scatter_(1, max_value_indexes, 1)
            hist_gs_set[:, i, :(i + 1)] = hist_gs

        return hist_gs_set


class INFER_MODULE(nn.Module):
    """docstring for INFER_MODULE"""

    def __init__(self, hparams):
        super(INFER_MODULE, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            GatedTrans(
                hparams.word_embedding_size,
                hparams.lstm_hidden_size),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=hparams.dropout_fc),
            nn.Linear(
                hparams.lstm_hidden_size,
                2
            )
        )

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques):

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = self.embed(ques)  # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_embed = F.normalize(ques_embed, p=2,
                                 dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_logits = self.att(ques_embed)  # shape: (batch_size, num_rounds, 2)

        logits = ques_logits.view(-1, 2)
        if self.training:
            ques_gs = F.gumbel_softmax(logits, hard=True)  # shape: (batch_size, i+1)
        else:
            _, max_value_indexes = logits.detach().max(1, keepdim=True)
            ques_gs = logits.detach().clone().zero_().scatter_(1, max_value_indexes, 1)
        ques_gs = ques_gs.view(-1, num_rounds, 2)

        Lambda = self.softmax(ques_logits)

        return ques_gs, Lambda


class RvA_MODULE(nn.Module):
    """docstring for R_CALL"""

    def __init__(self, hparams):
        super(RvA_MODULE, self).__init__()

        self.INFER_MODULE = INFER_MODULE(hparams)
        self.PAIR_MODULE = PAIR_MODULE(hparams)
        self.ATT_MODULE = ATT_MODULE(hparams)

    def forward(self, img, ques, hist):
        # img shape: [batch_size, num_proposals, i_dim]
        # img_att_ques shape: [batch_size, num_rounds, num_proposals]
        # img_att_cap shape: [batch_size, 1, num_proposals]
        # ques_gs shape: [batch_size, num_rounds, 2]
        # hist_logits shape: [batch_size, num_rounds, num_rounds]
        # ques_gs_prob shape: [batch_size, num_rounds, 2]

        cap_feat, ques_feat, ques_encoded = ques

        batch_size = ques_feat.size(0)
        num_rounds = ques_feat.size(1)
        num_proposals = img.size(1)

        ques_gs, ques_gs_prob = self.INFER_MODULE(ques_feat)  # (batch_size, num_rounds, 2)
        hist_gs_set = self.PAIR_MODULE(hist, ques_encoded)
        img_att_ques = self.ATT_MODULE(img, ques_feat)
        img_att_cap = self.ATT_MODULE(img, cap_feat)

        # soft
        ques_prob_single = torch.Tensor(data=[1, 0]).view(1, -1).repeat(batch_size, 1)  # shape: [batch_size, 2]
        ques_prob_single = ques_prob_single.cuda()
        ques_prob_single.requires_grad = False

        img_att_refined = img_att_ques.data.clone().zero_()  # shape: [batch_size, num_rounds, num_proposals]
        for i in range(num_rounds):
            if i == 0:
                img_att_temp = img_att_cap.view(-1, img_att_cap.size(-1))  # shape: [batch_size, num_proposals]
            else:
                hist_gs = hist_gs_set[:, i, :(i + 1)]  # shape: [batch_size, i+1]
                img_att_temp = torch.cat((img_att_cap, img_att_refined[:, :i, :]),
                                         dim=1)  # shape: [batch_size, i+1, num_proposals]
                img_att_temp = torch.sum(hist_gs.unsqueeze(-1) * img_att_temp,
                                         dim=-2)  # shape: [batch_size, num_proposals]
            img_att_cat = torch.cat((img_att_ques[:, i, :].unsqueeze(1), img_att_temp.unsqueeze(1)),
                                    dim=1)  # shape: [batch_size ,2, num_proposals]
            # soft
            ques_prob_pair = ques_gs_prob[:, i, :]
            ques_prob = torch.cat((ques_prob_single, ques_prob_pair), dim=-1)  # shape: [batch_size, 2]
            ques_prob = ques_prob.view(-1, 2, 2)  # shape: [batch_size, 2, 2]
            ques_prob_refine = torch.bmm(ques_gs[:, i, :].view(-1, 1, 2), ques_prob).view(-1, 1,
                                                                                          2)  # shape: [batch_size, num_rounds, 2]

            img_att_refined[:, i, :] = torch.bmm(ques_prob_refine, img_att_cat).view(-1,
                                                                                     num_proposals)  # shape: [batch_size, num_proposals]

        return img_att_refined, (ques_gs, hist_gs_set, img_att_ques)
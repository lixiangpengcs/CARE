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

class RegionGridCorrelation(nn.Module):
    def __init__(self, hparams):
        super(RegionGridCorrelation, self).__init__()
        self.hparams = hparams
        h=8
        self.linear_grid = nn.Linear(self.hparams.img_feature_size, hparams.hidden_size)
        self.linear_grid_dropout = nn.Dropout(0.10)
        self.grid_layernorm = nn.LayerNorm(hparams.hidden_size)
        self.linear_region = nn.Linear(self.hparams.img_feature_size, hparams.hidden_size)
        self.linear_region_dropout = nn.Dropout(0.10)
        self.region_layernorm = nn.LayerNorm(hparams.hidden_size)

        self.box_embedding = nn.Linear(4, 512)
        self.grid_embedding = PositionEmbeddingSine(256, normalize=True)

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])
        N = 3
        self.layers_region = nn.ModuleList([SelfAtt(d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                                                    identity_map_reordering=False,
                                                    attention_module=None,
                                                    attention_module_kwargs=None)
                                            for _ in range(N)])
        self.layers_grid = nn.ModuleList([SelfAtt(d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                                                    identity_map_reordering=False,
                                                    attention_module=None,
                                                    attention_module_kwargs=None)
                                          for _ in range(N)])

        self.region2grid = nn.ModuleList([LCCA(d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                                                    identity_map_reordering=False,
                                                    attention_module=None,
                                                    attention_module_kwargs=None)
                                          for _ in range(N)])

        self.grid2region = nn.ModuleList([LCCA(d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                                                    identity_map_reordering=False,
                                                    attention_module=None,
                                                    attention_module_kwargs=None)
                                          for _ in range(N)])


    def forward(self, batch):
        if self.hparams.grid_feat_trigger and self.hparams.grid_region_trigger:
            reg_img = batch['img_feat']
            reg_bbox = batch['region_bbox']
            aligns = batch['aligns'] # bs x num_obj x 49
            bs = aligns.size()[0]
            proj_reg_img = self.linear_region_dropout(F.relu(self.linear_region(reg_img)))
            proj_reg_img = self.region_layernorm(proj_reg_img)
            grid_feat = batch['grid_feat']
            grid_bbox = batch['grid_bbox']
            proj_grid_img = self.linear_grid_dropout(self.linear_grid(grid_feat))
            proj_grid_img = self.grid_layernorm(proj_grid_img)

            relative_positions = AllRelationalEmbedding(reg_bbox) # bs x (49+num_obj) x (49+num_obj) x 64
            flatten_relative_geometry_embeddings = relative_positions.view(-1, 64)
            box_size_per_head = list(relative_positions.shape[:3]) # bs, N, N
            box_size_per_head.insert(1, 1)
            relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l
                                                  in
                                                  self.WGs]

            attention_mask_region = (torch.sum(reg_img, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
            attention_mask_grid = (torch.sum(grid_feat, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

            relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
            relative_geometry_weights = F.relu(relative_geometry_weights)
            n_regions = proj_reg_img.shape[1]  # 100
            n_grids = proj_grid_img.shape[1]  # 49

            region2region = relative_geometry_weights[:, :, :n_regions, :n_regions]
            grid2grid = relative_geometry_weights[:, :, n_regions:, n_regions:]
            region2all = relative_geometry_weights[:, :, :n_regions, :]
            grid2all = relative_geometry_weights[:, :, n_regions:, :]

            aligns = aligns.unsqueeze(1) # bs x 1 x regions x n_grid
            tmp_mask = torch.eye(n_regions, device=reg_img.device).unsqueeze(0).unsqueeze(0)
            tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_regions * n_regions
            region_aligns = (torch.cat([tmp_mask, aligns], dim=-1) == 0)  # bs * 1 * n_regions *(n_regions+n_grids)

            tmp_mask = torch.eye(n_grids, device=reg_img.device).unsqueeze(0).unsqueeze(0)
            tmp_mask = tmp_mask.repeat(bs, 1, 1, 1)  # bs * 1 * n_grids * n_grids
            grid_aligns = (torch.cat([aligns.permute(0, 1, 3, 2), tmp_mask],
                                     dim=-1) == 0)  # bs * 1 * n_grids *(n_grids+n_regions)

            reg_box_embeded = self.box_embedding(reg_bbox) # bs x obj_num x 512
            grid_box_embeded = self.grid_embedding(grid_bbox) # bs x 16
            pos_cross = torch.cat([reg_box_embeded, grid_box_embeded], dim=-2) # bs x 149 x 512

            out_grid = proj_grid_img
            out_region = proj_reg_img

            for l_region, l_grid, l_r2g, l_g2r in zip(self.layers_region, self.layers_grid, self.region2grid,
                                                      self.grid2region):
                out_region = l_region(out_region, out_region, out_region, region2region, attention_mask_region, pos=reg_box_embeded)
                out_grid = l_grid(out_grid, out_grid, out_grid, grid2grid, attention_mask_grid, pos=grid_box_embeded)
                out_all = torch.cat([out_region, out_grid], dim=1) # bs x 149 x 512

                out_region = l_r2g(out_region, out_all, out_all, region2all, region_aligns, pos_source=reg_box_embeded, pos_cross=pos_cross)
                # print('grid cross')
                out_grid = l_g2r(out_grid, out_all, out_all, grid2all, grid_aligns, pos_source=grid_box_embeded, pos_cross=pos_cross)

            out = torch.cat([out_region, out_grid], dim=1)
            attention_mask = torch.cat([attention_mask_region, attention_mask_grid], dim=-1)
            return out, attention_mask.squeeze(1).squeeze(1)


        elif self.hparams.grid_region_trigger:
            reg_img = batch['img_feat']
            proj_reg_img = self.linear_region_dropout(F.relu(self.linear_region(reg_img)))
            img = self.region_layernorm(proj_reg_img)
        else:
            img = batch['img_feat']
            img = self.linear_region_dropout(self.linear_region(img))
            """image feature normarlization"""
            if self.hparams.img_norm:
                img = F.normalize(img, dim=1, p=2)
        mask = (0 != img.abs().sum(-1))
        return img, mask

    def get_pos_embedding(self, boxes, grids, split=False):
        bs = boxes.shape[0]
        region_embed = self.box_embedding(boxes)
        grid_embed = self.grid_embedding(grids.view(bs, 7, 7, -1))
        if not self.args.box_embed:
            # print('reach here')
            region_embed = torch.zeros_like(region_embed)
        if not self.args.grid_embed:
            # print('reach here')
            grid_embed = torch.zeros_like(grid_embed)
        if not split:
            pos = torch.cat([region_embed, grid_embed], dim=1)
            return pos
        else:
            return region_embed, grid_embed

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


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask=None):
        bs = x.size(0)
        x = x.view(bs, 7, 7, 4)
        if mask is None:
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        not_mask = (mask == False)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        pos = pos.flatten(1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding

def BoxRelationEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    batch_size = f_g.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
    cx = (x_min+x_max) * 0.5
    cy = (y_min+y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat([delta_x, delta_y, delta_w, delta_h], -1)

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)

def GridRelationalEmbedding(batch_size, grid_size=7, dim_g=64, wave_len=1000, trignometric_embedding=True):
    # make grid
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(batch_size, -1, -1)
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


def get_normalized_grids(bs, grid_size=7):
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(bs, -1, -1) / grid_size
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)
    return y_min, x_min, y_max, x_max

def AllRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True, require_all_boxes=False):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image
    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j
    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
    grid_x_min, grid_y_min, grid_x_max, grid_y_max = get_normalized_grids(batch_size)

    x_min = torch.cat([x_min, grid_x_min], dim=1)
    y_min = torch.cat([y_min, grid_y_min], dim=1)
    x_max = torch.cat([x_max, grid_x_max], dim=1)
    y_max = torch.cat([y_max, grid_y_max], dim=1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda() # [0., 1., 2., 3., 4., 5., 6., 7.]
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat # bs x N x N x 4 x 8
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1) # bs x N x N x 32
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    if require_all_boxes:
        all_boxes = torch.cat([x_min, y_min, x_max, y_max], dim=-1)
        return (embedding), all_boxes
    return (embedding) # bs x N x N x 64

class PositionWiseFeedForward(nn.Module):
    '''
    Position-wise feed forward layer
    '''

    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out

class ScaledDotProductWithBoxAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductWithBoxAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            pad_attention_mask = ~attention_mask
            att = att.masked_fill(pad_attention_mask, -10000.0)

        w_g = box_relation_embed_matrix
        w_a = att

        w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        w_mn = torch.softmax(w_mn, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithBoxAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values,box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm,box_relation_embed_matrix, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values,box_relation_embed_matrix, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

class SelfAtt(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(SelfAtt, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        # self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
        #                                 attention_module=attention_module,
        #                                 attention_module_kwargs=attention_module_kwargs)
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos=None):
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

class LCCA(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(LCCA, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos_source=None, pos_cross=None):
        q = queries + pos_source
        k = keys + pos_cross
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff

if __name__ == '__main__':
    x = torch.rand((16, 32, 4)).float().cuda()
    pos = AllRelationalEmbedding(x)
    print(pos.shape)
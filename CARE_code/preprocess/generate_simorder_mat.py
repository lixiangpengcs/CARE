import h5py
import pickle as pkl
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import json
import tqdm
import numpy as np
import torch
from transformers import BertTokenizer
from transformers import BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').cuda()
from collections import Counter

ans_maxlen = 10
mode = 'val'

positive_phrases = ['yes', 'yeah', 'yyes', 'ye', 'i think so']
negative_phrases = ['no', 'nope', 'not really', 'i don\'t think so', 'no it\'s not']
pos_flag = False
neg_flag = False

output_path = '/home/lixiangpeng/workspace/vis_dialog/MVAN-VisDial/cached_data/CL_data/'

def convert_str_to_idx_withpadding(ans):
    ans_tokens = tokenizer.tokenize(ans)
    if len(ans_tokens) > ans_maxlen-2:
        ans_tokens = ans_tokens[:(ans_maxlen-2)]
    ans_tokens = ['[CLS]'] + ans_tokens + ['[SEP]']
    token_idxs = tokenizer.convert_tokens_to_ids(ans_tokens)
    ans_mask = [1] * len(ans_tokens)
    while len(token_idxs) < ans_maxlen:
        token_idxs.append(0)
        ans_mask.append(0)
    token_tensor = token_idxs
    token_mask = ans_mask
    return token_tensor, token_mask
def similarity_computation(gt_idx, opts):
    opt_idxs = []
    opt_masks = []
    for opt in opts:
        opt_idx, opt_mask = convert_str_to_idx_withpadding(opt)
        opt_idxs.append(opt_idx)
        opt_masks.append(opt_mask)
    opt_tensors = torch.tensor(opt_idxs).cuda()
    opt_masks = torch.tensor(opt_masks).cuda()
    with torch.no_grad():
        out = model(opt_tensors, opt_masks)
        pool_out = torch.mean(out['last_hidden_state'], dim=1)
        ans_embedding = pool_out[gt_idx]
        extend_ans_embedding = ans_embedding.unsqueeze(0)
        # cos similarity
        similarity_scores = torch.cosine_similarity(extend_ans_embedding, pool_out)

    return similarity_scores.cpu()

if __name__ == '__main__':
    data_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/visdial_1.0_img'
    img2objnum = {}
    with h5py.File(os.path.join(data_path, 'features_dan_faster_rcnn_x101_%s.h5' % mode), 'r') as f:
        for key in f.keys():
            print(key)
        features = f['image_features'][:]
        image_bb = f['image_bb'][:]
        pos_boxes = f['pos_boxes'][:]
        spatial_features = f['spatial_features'][:]
        print(type(features))

    with open(os.path.join(data_path, '%s_imgid2idx.pkl' % mode), 'rb') as f:
        data = pkl.load(f)
        print(len(data.keys()))
        data_keys = data.keys()

    num_counter = []
    for i, img_id in enumerate(data_keys):
        idx = data[img_id]
        box_num = pos_boxes[idx]
        # print(img_id, box_num)
        img2objnum[img_id] = box_num[-1] - box_num[0] - 1
        num_counter.append(box_num[-1] - box_num[0] - 1)

    counter = Counter(num_counter)
    with open('/home/lixiangpeng/data/dataset/visual_dialog/data/v1.0/visdial_1.0_%s.json'%mode, 'r') as f:
        ori_data = json.load(f)
        data = ori_data['data']
        dialogs = data['dialogs']
        questions = data['questions']
        answers = data['answers']

    ans_opts = []

    clidx = 0
    imgid2clidx = {}
    cl_mat = []

    for dialog in tqdm.tqdm(dialogs):
        dia = dialog['dialog']
        img_id = dialog['image_id']
        cl_dia_mat = []
        for qa in dia:
            ans_opt = {}
            ans_opt['id'] = id
            ans_opt['img_id'] = img_id
            ans_opt['gt_idx'] = qa['gt_index']
            ans_opt['answer'] = answers[qa['answer']]
            options = qa['answer_options']
            ans_opt['options'] = [answers[i] for i in options]
            ans_opt['objnum'] = img2objnum[img_id]
            ques = questions[qa['question']]
            ans_opt['ques_len'] = len(ques.split())
            similarity_scores = similarity_computation(ans_opt['gt_idx'], ans_opt['options'])
            ans_opt['similarity_scores'] = similarity_scores
            (a, order) = torch.sort(similarity_scores, descending=True)

            ans_opt['order'] = order.numpy()
            cl_dia_mat.append(order.numpy())
            ans_opts.append(ans_opt)

        cl_mat.append(np.array(cl_dia_mat))
        imgid2clidx[img_id] = clidx
        clidx += 1

    with open(output_path+'%s_imgid2clidx.pkl' % mode, 'wb') as f:
        pkl.dump(imgid2clidx, f)

    np.save(output_path+'%s_simordermat.npy' % mode, np.array(cl_mat))
'''
generate the similarity orders by glove embedding
'''

import numpy as np
import h5py
import os
import pickle as pkl
import tqdm
import json
import torch

mode = 'train'
output_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/CL_data/'
MAX_ANSWER_LEN = 10

special_tokens = ['(', ')', ',', '.', '[', ']', '?', '!', '%', '-', '---', '&', '*']

class GloveProcessor(object):
  def __init__(self, glove_path='/home/lixiangpeng/data/models/glove/glove.6B.300d.txt'):
    self.glove_path = glove_path

  def _load_glove_model(self):
    print("Loading pretrained word vectors...")
    with open(self.glove_path, 'r') as f:
      model = {}
      for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])  # e.g., 300 dimension
        model[word] = embedding

    print("Done.", len(model), " words loaded from %s" % self.glove_path)

    return model

gp = GloveProcessor()
gp_model = gp._load_glove_model()

def get_gloveembedding(opt):

    sent = opt.lower()
    for item in special_tokens:
        sent = sent.replace(item, "")
    opt_tokens = sent.replace("'s", " 's").strip().split()
    opt_len = len(opt_tokens)
    opt_vectors = []
    # assert opt_len>0
    if opt_len>0:
        for opt_token in opt_tokens:
            if opt_token in gp_model:
                opt_vector = gp_model[opt_token]
            else:
                opt_vector = gp_model['unk']
            opt_vectors.append(opt_vector)
        opt_vectors_np = np.array(opt_vectors)
        opt_vectors_avg = np.mean(opt_vectors_np, axis=0)
    else:
        opt_vectors_avg = np.zeros(300)
    return opt_vectors_avg

def similarity_computation(gth_idx, opts):
    opt_idxs = []
    opt_masks = []
    for opt in opts:
        opt_vector = get_gloveembedding(opt)
        opt_idxs.append(opt_vector)
    opt_idx_np = torch.tensor(np.array(opt_idxs))
    gth_vector = torch.tensor(opt_idx_np[gth_idx]).unsqueeze(0)
    similarity_scores = torch.cosine_similarity(gth_vector, opt_idx_np)
    return similarity_scores

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

    sorted_options = []
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

            # print('GTH: ', ans_opt['options'][qa['gt_index']])
            # for i in range(100):
            #     sorted_options.append(ans_opt['options'][order[i]])
            #     print(i, ':\t', ans_opt['options'][order[i]])
        cl_mat.append(np.array(cl_dia_mat))
        imgid2clidx[img_id] = clidx
        clidx += 1
    with open(output_path+'%s_imgid2clidx_byglove.pkl' % mode, 'wb') as f:
        pkl.dump(imgid2clidx, f)

    np.save(output_path+'%s_simordermat_byglove.npy' % mode, np.array(cl_mat))
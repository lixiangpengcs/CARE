import torch
import numpy as np
import pickle as pkl
import xlwt
from collections import  Counter
import json

mode = 'train'
data_path = '/home/lixiangpeng/workspace/vis_dialog/MVAN-VisDial/preprocess/%s_simorder_file.pkl'%mode

with open('/raid/lixiangpeng/dataset/visual_dialog/data/v1.0/visdial_1.0_%s.json'%mode, 'r') as f:
    ori_data = json.load(f)
data = ori_data['data']
dialogs = data['dialogs']
questions = data['questions']
answers = data['answers']
with open('%s_lengthstatistic.pkl'%mode, 'rb') as f:
    result = pkl.load(f)
    key_list = list(result.keys())
    key_list.sort()
    # print(key_list)
    for key in key_list:
        print(key, ' : ', result[key])
with open('%s_lens2imgid'%mode, 'rb') as f:
    len2imgid = pkl.load(f)
    print(len2imgid[126])
    imgid = len2imgid[126][0]

    for dia in dialogs:
        if dia['image_id'] == imgid:
            dialog = dia['dialog']

            for qa in dialog:
                print(questions[qa['question']])
                print(answers[qa['answer']])



import torch
import numpy as np
import pickle as pkl
import xlwt
from collections import Counter
mode = 'train'
data_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/CL_data/%s_simorder_file.pkl'%mode
output_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/CL_preprocess/'

dia_cnt = 0
dia_ques_lens = []
dia_img_lens = []
dia_ans_lens = []

imgid2len = {}
with open(data_path, 'rb') as f:
    data = pkl.load(f)
    inner_cnt = 0
    ques_lens = []
    img_lens = []
    ans_lens = []
    for i, ins in enumerate(data):

        img_lens.append(ins['objnum'])
        ques_lens.append(ins['ques_len'])
        ans_lens.append(len(ins['answer'].split()))
        inner_cnt += 1

        if inner_cnt % 10 == 0:
            dia_ques_lens.append(ques_lens)
            dia_img_lens.append(img_lens)
            dia_ans_lens.append(ans_lens)
            imgid2len[ins['img_id']] = sum(ques_lens) + sum(img_lens) + sum(ans_lens)
            ques_lens = []
            img_lens = []
            ans_lens = []

ques_lens = np.array(dia_ques_lens)
dia_img_lens = np.array(dia_img_lens)
ans_lens = np.array(dia_ans_lens)
tol_lens = ques_lens + dia_img_lens + ans_lens
sum_lens = tol_lens.sum(axis=-1)
sum_lens_list = sum_lens.tolist()
result = Counter(sum_lens_list)
# print(result)
# print(list(result.keys()))
key_list = list(result.keys())
key_list.sort()
# print(key_list)
for key in key_list:
    print(key, ' : ', result[key])


workbook = xlwt.Workbook(encoding = 'utf-8')
worksheet = workbook.add_sheet('sheet')
for i, key in enumerate(key_list):
    worksheet.write(i, 0, int(key))
    worksheet.write(i, 1, int(result[key]))
workbook.save(output_path+'visualization_%s_iqa.xls'%mode)
with open(output_path+'%s_lengthstatistic_iqa.pkl'%mode, 'wb') as f:
    pkl.dump(dict(result), f)

len2img_id = {}
for imgid in imgid2len.keys():
    idlen = imgid2len[imgid]
    if idlen not in len2img_id.keys():
        len2img_id[idlen] = [imgid]
    else:
        len2img_id[idlen].append(imgid)
print('done')

with open(output_path+'%s_lens2imgid_iqa.pkl'%mode, 'wb') as f:
    pkl.dump(len2img_id, f)

print(len2img_id[126])
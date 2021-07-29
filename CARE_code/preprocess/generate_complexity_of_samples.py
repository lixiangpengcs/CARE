import numpy as np
import pickle as pkl
import xlwt
from collections import Counter

mode = 'train'
modes = ['train', 'val']
data_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/CL_data/%s_simorder_file.pkl'%mode
output_path = './cached_data/CL_data/'

dia_ques_lens = []
dia_img_lens = []
dia_ans_lens = []

if __name__ == '__main__':
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
                imgid2len[ins['img_id']] = sum(ques_lens) + sum(img_lens)
                ques_lens = []
                img_lens = []
                ans_lens = []
    ques_lens = np.array(dia_ques_lens)
    dia_img_lens = np.array(dia_img_lens)
    ans_lens = np.array(dia_ans_lens)
    tol_lens = ques_lens + dia_img_lens
    sum_lens = tol_lens.sum(axis=-1)
    sum_lens_list = sum_lens.tolist()
    lens2number = Counter(sum_lens_list)
    lens_list = list(lens2number.keys())
    lens_list.sort()

    cfd = [] # cumulative Fraction Data
    cfd_sum = 0
    number_list = []
    lens2weight = {}
    lens2cfdweight = {}
    cfd_a = 0.0

    temp_tol = 0
    for length in lens_list:
        temp_tol += lens2number[length]
    print(temp_tol)

    for length in lens_list:
        number_list.append(lens2number[length])
        lens2weight[length] = lens2number[length] / temp_tol
        cfd_sum += lens2number[length]
        cfd.append(cfd_sum)
        lens2cfdweight[length] = cfd_sum / temp_tol
    tol_len = cfd[-1]
    cfd = (np.array(cfd) / tol_len).tolist()

    # len2imgid = {}
    # for imgid in imgid2len.keys():
    #     idlen = imgid2len[imgid]
    #     if idlen not in len2imgid.keys():
    #         len2imgid[idlen] = [imgid]
    #     else:
    #         len2imgid[idlen].append(imgid)
    # print('done')

    # imgid2lens = {}
    # for ins_len in lens_list:
    #     imgids = len2imgid[ins_len]
    #     for imgid in imgids:
    #        imgid2lens[imgid] = lens2weight[ins_len]

    imgid2weight = {}
    imgid2cfdweight = {}
    for imgid in imgid2len.keys():
        length = imgid2len[imgid]
        imgid2weight[imgid] = lens2weight[length]
        imgid2cfdweight[imgid] = lens2cfdweight[length]
    print('done')
    # with open(output_path+'%s_imgid2weight.pkl'%mode, 'wb') as f:
    #     pkl.dump(imgid2weight, f)

    # with open(output_path+'%s_imgid2cfdweight_iq.pkl'%mode, 'wb') as f:
    #     pkl.dump(imgid2cfdweight, f)
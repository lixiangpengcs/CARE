import numpy as np
import xlwt
import pickle as pkl
data_path = '/home/lixiangpeng/workspace/vis_dialog/MVAN-VisDial/preprocess/output_data/'

lens2weight = {}

mode='train'
with open(data_path+'%s_lengthstatistic.pkl'%mode, 'rb') as f:
    result = pkl.load(f)

x_list = list(result.keys())
x_list.sort()

cfd = []
cfd_sum = 0
y_list = []

tol_len = 123287
for x in x_list:
    y_list.append(result[x])
    cfd_sum += result[x]
    cfd.append(cfd_sum)
    lens2weight[x] = cfd_sum/tol_len
tol_len = cfd[-1]

cfd = (np.array(cfd) / tol_len).tolist()

imgid2lens = {}

with open(data_path+'%s_lens2imgid.pkl'%mode, 'rb') as f:
    len2imgid = pkl.load(f)
    lens = len2imgid.keys()
    for ins_len in lens:
        imgids = len2imgid[ins_len]
        for imgid in imgids:
           imgid2lens[imgid] = lens2weight[ins_len]

print('done!')
with open(data_path+'%s_imgid2weight.pkl'%mode, 'wb') as f:
    pkl.dump(imgid2lens, f)
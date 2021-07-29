import torch
import h5py
import os

import sys
mode = 'val'
root_path = '/raid/lixiangpeng/dataset/visual_dialog/data/visdial_1.0_text'
with h5py.File(os.path.join(root_path, 'visdial_1.0_multi_text_%s.hdf5'%mode)) as text_features:
    data_keys = text_features.keys()
    print(data_keys)
    ques = text_features['ques']
    img_ids = text_features['img_ids']
    print(img_ids[:10])
    print('done')

sys.exit()
import h5py
import pickle as pkl
import os
import sys

mode = 'val'

img2objnum = {}

data_path = '/raid/lixiangpeng/dataset/visual_dialog/data/visdial_1.0_img'

with h5py.File(os.path.join(data_path,'features_dan_faster_rcnn_x101_%s.h5'%mode), 'r') as f:
    for key in f.keys():
        print(key)
    features = f['image_features'][:]
    image_bb = f['image_bb'][:]
    pos_boxes = f['pos_boxes'][:]
    spatial_features = f['spatial_features'][:]
    print(type(features))
    # ex = pos_boxes[0]
    # ex2 = pos_boxes[1]
    # print(ex, ex2)


with open(os.path.join(data_path, '%s_imgid2idx.pkl'%mode), 'rb') as f:
    data = pkl.load(f)
    print(len(data.keys()))
    data_keys = data.keys()

for i, img_id in enumerate(data_keys):
    idx = data[img_id]
    box_num = pos_boxes[idx]
    # print(img_id, box_num)
    img2objnum[img_id] = box_num[-1] - box_num[0] -1
    #sys.exit()

with open(os.path.join(data_path, '%s_img2objnum.pkl'%mode), 'wb') as f:
    pkl.dump(img2objnum, f)
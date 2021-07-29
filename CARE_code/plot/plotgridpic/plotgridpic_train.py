
import json
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle as pkl

mode = 'train'
f = open('/raid/lixiangpeng/dataset/visual_dialog/data/v1.0/visdial_1.0_%s.json'%mode, 'r')
ori_data = json.load(f)
data = ori_data['data']
print(data.keys())

dialogs = data['dialogs']
questions = data['questions']
answers = data['answers']

# imgid = '57205'   # floor
# imgid = '42067'    # door
# imgid = '21734'
imgid = '10701'
K=5

imgid2path_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/visdial_1.0_img/train_imgid2imgpath.pkl'
imgid2path = pkl.load(open(imgid2path_path, 'rb'))

def construct_obj_dict():
    with open('/home/lixiangpeng/workspace/feat_ext/grid-feats-vqa/datasets/vg/1600-400-20/objects_vocab.txt', 'r') as f:
        lines = f.readlines()
        obj_dict = []
        for i, obj in enumerate(lines):
            obj_dict.append(obj.strip('\n'))
    return obj_dict

str_len = len(imgid)
img_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/original_imgs/' + imgid2path[int(imgid)]
print(img_path)
im = cv2.imread(img_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.xticks([])
plt.yticks([])
feature_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/grid_feat/region_feats/visdial/train/%s.npy'%imgid
feature = np.load(feature_path, allow_pickle=True).item()
bboxs = feature['bbox'][:K]
objects = feature['objects'][:K]
obj_dict = construct_obj_dict()
for i, bbox in enumerate(bboxs):
    plt.gca().add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='blue', linewidth=3, alpha=0.5)
    )
    plt.gca().text(bbox[0], bbox[1] - 2,
                   '%s' % (str(i)+'-'+obj_dict[objects[i]]),
                   bbox=dict(facecolor='blue', alpha=0.5),
                   fontsize=10, color='white')
x_cords = []
y_cords = []
image_height = feature['image_height']
image_width = feature['image_width']
# x_grid_size = image_width / 7
# y_grid_size = image_height / 7
# x_init = 0
# y_init = 0
# for i in range(6):
#     x_init += x_grid_size
#     x_cords.append(x_init)
#     y_init += y_grid_size
#     y_cords.append(y_init)
#
# Xs = []
# Ys = []
# for i, x in enumerate(x_cords):
#     Xs.append([x, 0])
#     Ys.append([x, image_height])
# for j in y_cords:
#     Xs.append([0, j])
#     Ys.append([image_width, j])
# for a in range(len(Xs)):
#     m = Xs[a]
#     n = Ys[a]
#     plt.plot(m, n, color='r')
grid_size = 7
h,w = im.shape[0:2]
delta_w = w / grid_size
delta_h = h / grid_size
grid_bbox = []

a = [[3,4], [4,4], [5,4],
     [3,5], [4,5], [5,5],
     [3,6], [4,6], [5,6]]


# for i in range(0,grid_size):
#     pos_w = delta_w * i
#     for j in range(0, grid_size):
#         pos_h = delta_h * j
#         if [i, j] in a:
#             plt.gca().add_patch(
#                 plt.Rectangle((pos_w, pos_h),
#                               delta_w,
#                               delta_h, fill='purple',
#                               edgecolor='red', linewidth=1, alpha=0.5)
#             )
#         if i>0 and i<6 and j>0 and j<4:
#             plt.gca().add_patch(
#                 plt.Rectangle((pos_w, pos_h),
#                               delta_w,
#                               delta_h, fill='purple',
#                               edgecolor='red', linewidth=1, alpha=0.5)
#             )
#         else:
#             plt.gca().add_patch(
#                 plt.Rectangle((pos_w, pos_h),
#                               delta_w,
#                               delta_h, fill=False,
#                               edgecolor='red', linewidth=1, alpha=0.5)
#             )
for i in range(0,grid_size):
    pos_w = delta_w * i
    for j in range(0, grid_size):
        pos_h = delta_h * j
        plt.gca().add_patch(
                plt.Rectangle((pos_w, pos_h),
                              delta_w,
                              delta_h, fill=False,
                              edgecolor='red', linewidth=1, alpha=0.5)
            )
plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
plt.margins(0, 0)
# plt.savefig('test.jpg')
plt.show()
print('done')

import json
import os
import sys
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

mode = 'val'
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
imgid = '424808'
K=10

def construct_obj_dict():
    with open('/home/lixiangpeng/workspace/feat_ext/grid-feats-vqa/datasets/vg/1600-400-20/objects_vocab.txt', 'r') as f:
        lines = f.readlines()
        obj_dict = []
        for i, obj in enumerate(lines):
            obj_dict.append(obj.strip('\n'))
    return obj_dict

gt_caption = 'a boy in jeans and a white tee shirt jumping over a cone on a skateboard'
# gt_caption = 'a clown holding a colorful umbrella with costume on him'
# for dialog in dialogs:
#     dia = dialog['dialog']
#     caption = dialog['caption']
#     if caption == gt_caption:
#         print(dialog['image_id'])
#         for i, d in enumerate(dia):
#             ques = questions[d['question']]
#             ans = answers[d['answer']]
#             print(i)
#             print(ques)
#             print(ans)

str_len = len(imgid)
img_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/original_imgs/V_1.0_val_image/VisualDialog_val2018/VisualDialog_val2018_'+'0'*(12-str_len) +'%s.jpg'%imgid
im = cv2.imread(img_path)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)
plt.xticks([])
plt.yticks([])
feature_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/grid_feat/region_feats/visdial/val/%s.npy'%imgid
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
plt.savefig('test.jpg')
plt.show()
print('done')
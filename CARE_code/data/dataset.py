from typing import Any, Dict, List, Optional

import torch
import h5py
import pickle
import json
import os
import numpy as np
from torch.utils.data import Dataset

from data.preprocess.init_glove import Vocabulary

class VisDialDataset(Dataset):
  """
  A full representation of VisDial v1.0 (train/val/test) dataset. According
  to the appropriate split, it returns dictionary of question, image,
  history, ground truth answer, answer options, dense annotations etc.
  """

  def __init__(
      self,
      hparams,
      overfit: bool = False,
      split: str = "",
      old_split=None,
  ):
    super().__init__()
    self.hparams = hparams

    self.split = split
    self.vocabulary = Vocabulary(hparams.word_counts_json, min_count=hparams.vocab_min_count)

    # train, val, test
    text_features_hdfpath = hparams.text_features_h5 % (self.hparams.model_train_type, self.split)
    img_features_h5_path = hparams.img_features_h5 % (self.hparams.img_feature_type, self.split)

    self.hdf_reader = DataHdfReader(hparams, text_features_hdfpath, img_features_h5_path, self.split, old_split)

    # Keep a list of image_ids as primary keys to access data.
    self.text_feat_image_ids = list(self.hdf_reader.text_feature_id_l)
    print("image ids", len(self.text_feat_image_ids))
    if overfit:
      self.text_feat_image_ids = self.text_feat_image_ids[:5]
    self.float_variables = ["img_feat", "gt_relevance", 'sample_weight', 'grid_feat', 'region_bbox', 'aligns', 'grid_feat',
                            'grid_bbox']

  def __len__(self):
    return len(self.text_feat_image_ids)

  def __getitem__(self, index):
    # Get image_id, which serves as a primary key for current instance.
    # Get image features for this image_id using hdf reader.
    curr_features = self.hdf_reader[index]

    for f_key in curr_features.keys():
      if f_key in self.float_variables:
        curr_features[f_key] = torch.tensor(curr_features[f_key]).float()
        continue
      curr_features[f_key] = torch.tensor(curr_features[f_key]).long()

    return curr_features

  def collate_fn(self, batch):
    merged_batch = {key: [d[key] for d in batch] for key in batch[0]}
    max_np = max(merged_batch["num_proposals"])
    for key in merged_batch:
      if key in ['img_feat', 'region_bbox', 'aligns']:
        for batch_idx, features in enumerate(merged_batch[key]):
          pad_features = torch.zeros((max_np - len(features), features.size()[1])).float()
          merged_batch[key][batch_idx] = torch.cat((features, pad_features), axis=0)

      merged_batch[key] = torch.stack(merged_batch[key], 0)

    return merged_batch


class VisualDialogOldVersion(object):
  def __init__(self):
    pass

  def get_train_img_ids(self, train_jsonpath):
    with open(train_jsonpath, "r") as visdial_file:
      visdial_data = json.load(visdial_file)
      train_img_ids = [dialog_for_image["image_id"] for dialog_for_image in visdial_data["data"]["dialogs"]]

    return train_img_ids

  def get_val_img_ids(self, val_jsonpath):
    with open(val_jsonpath, "r") as visdial_file:
      visdial_data = json.load(visdial_file)
      val_img_ids = [dialog_for_image["image_id"] for dialog_for_image in visdial_data["data"]["dialogs"]]

    return val_img_ids


class DataHdfReader(object):
  """
  A reader for HDF files containing pre-extracted image features. A typical HDF file is expected
  to have a column named "image_id", and another column named "features".

  Example of an HDF file:
  ```
  visdial_train_faster_rcnn_bottomup_features.h5
     |--- "image_id" [shape: (num_images, )]
     |--- "features" [shape: (num_images, num_proposals, feature_size)]
     +--- .attrs ("split", "train")
  ```
  Refer ``$PROJECT_ROOT/data/extract_bottomup.py`` script for more details about HDF structure.

  Parameters
  ----------
  features_hdfpath : str
      Path to an HDF file containing VisDial v1.0 train, val or test split image features.
  in_memory : bool
      Whether to load the whole HDF file in memory. Beware, these files are sometimes tens of GBs
      in size. Set this to true if you have sufficient RAM - trade-off between speed and memory.
  """

  def __init__(self, hparams, text_features_h5_path: str, img_features_h5_path: str, split=None, old_split=None):

    self.text_features_h5_path = text_features_h5_path
    self.img_features_h5_path = img_features_h5_path

    self._split = split
    self.hparams = hparams
    # text
    with h5py.File(self.text_features_h5_path, "r") as text_features_h5:
      self.feature_keys = list(text_features_h5.keys())
      print("feature_keys", self.feature_keys)
      self._split = split
      assert split == self._split
      print("data split :", self._split)

      # visdial 0.9 or 1.0
      if self.hparams.dataset_version == '0.9':
        self.text_feature_id_l = self.get_old_img_ids(self.hparams.visdial_json % old_split)
        self.train_text_feature_id_set = set(self.get_old_img_ids(self.hparams.visdial_json % "train"))
        self.val_text_feature_id_set = set(self.get_old_img_ids(self.hparams.visdial_json % "val"))
        print(self.hparams.visdial_json % old_split)
        self.text_feature_h5_id_l = list(text_features_h5["img_ids"])
        self.old_split = old_split

      else:
        self.text_feature_id_l = list(text_features_h5["img_ids"])

    # image
    if hparams.img_feature_type == "dan_faster_rcnn_x101":
      # get imgid2id dicionary
      self.img_feature_id_l = list(pickle.load(open(hparams.imgid2idx_path % split, "rb")))
      with h5py.File(self.img_features_h5_path, "r") as features_hdf:
        self.pos_boxes = np.array(features_hdf.get("pos_boxes"))
    else:
      with h5py.File(self.img_features_h5_path, "r") as img_features_h5:
        self.img_feature_id_l = list(img_features_h5["image_id"])

    if hparams.grid_feat_trigger:
      with h5py.File(hparams.grid_feature_h5%self._split, "r") as grid_hdf:
        self.grid_features = np.array(grid_hdf['grid_features'])

    # curriculum trigger
    if hparams.curriculum_trigger and self._split=='train':
      print(" Curriculum Learning is adopt in this training stage!")
      if hparams.byglove_trigger:
        self.imgid2clidx = pickle.load(open(hparams.imgid2clidx_byglove_path, 'rb'))
        self.cl_data = np.load(hparams.cl_byglove_data_path)
      else:
        self.imgid2clidx = pickle.load(open(hparams.imgid2clidx_path, 'rb'))
        self.cl_data = np.load(hparams.cl_data_path)
      self.imgid2weights = pickle.load(open(hparams.imgid2weight_path, 'rb'))
      print('done')

  def __len__(self):
    return len(self.text_feature_id_l)

  def __getitem__(self, index: int):

    features = {}
    text_feature_index = index
    features['index'] = index

    if self.hparams.dataset_version == '0.9':
      image_id = self.text_feature_id_l[index]
      text_feature_index = self.text_feature_h5_id_l.index(image_id)

      if self.old_split == "train":
        assert image_id in self.train_text_feature_id_set
      elif self.old_split == "val":
        assert image_id in self.val_text_feature_id_set

    # text
    with h5py.File(self.text_features_h5_path, "r") as text_features_hdf:
      for f_key in self.feature_keys:
        features[f_key] = text_features_hdf[f_key][text_feature_index]
      image_id = text_features_hdf["img_ids"][text_feature_index]

    assert image_id == self.text_feature_id_l[index]

    # curriculum setting
    if self.hparams.curriculum_trigger and self._split=='train':
      cl_idx = self.imgid2clidx[image_id] # curriculum index
      cl_insdata = self.cl_data[cl_idx]
      weights = self.imgid2weights[image_id]
      assert weights >= 0.0 and weights<=1.0
      features['cl_order'] = cl_insdata # 10 x 100
      features['sample_weight'] = weights

    # image
    img_feature_index = self.img_feature_id_l.index(image_id)  # text / img index same??

    if self.hparams.img_feature_type == "dan_faster_rcnn_x101":
      if self.hparams.grid_region_trigger:
        image_features = np.load(os.path.join(self.hparams.grid_region_path%self._split, str(int(image_id))+".npy"), allow_pickle=True).item()
        features['img_feat'] = image_features['features']
        img_w = image_features['image_width']
        img_h = image_features['image_height']
        bbox = image_features['bbox']
        bbox[:, 0] = bbox[:, 0] / img_w
        bbox[:, 2] = bbox[:, 2] / img_w
        bbox[:, 1] = bbox[:, 1] / img_h
        bbox[:, 3] = bbox[:, 3] / img_h
        features['region_bbox'] = bbox
        features['aligns'] = image_features['aligns']
        features['num_proposals'] = image_features['features'].shape[0]
        assert image_features['features'].shape[0]==image_features['bbox'].shape[0]
      else:
        with h5py.File(self.img_features_h5_path, "r") as features_hdf:
          image_features = features_hdf["image_features"][self.pos_boxes[img_feature_index][0]: self.pos_boxes[img_feature_index][1], :]
          features["img_feat"] = image_features
          features["num_proposals"] = len(image_features)
      if self.hparams.grid_feat_trigger:
        grid_feat = self.grid_features[img_feature_index]
        features['grid_feat'] = grid_feat
        image_features = np.load(os.path.join(self.hparams.grid_region_path % self._split, str(int(image_id)) + ".npy"),
                                 allow_pickle=True).item()
        img_w = image_features['image_width']
        img_h = image_features['image_height']
        grid_boxes = self.get_grid_bboxes(img_w, img_h)
        grid_boxes[:, 0] = grid_boxes[:, 0] / img_w
        grid_boxes[:, 2] = grid_boxes[:, 2] / img_w
        grid_boxes[:, 1] = grid_boxes[:, 1] / img_h
        grid_boxes[:, 3] = grid_boxes[:, 3] / img_h
        features['grid_bbox'] = grid_boxes
    else:
      with h5py.File(self.img_features_h5_path, "r") as img_features_hdf:
        features["img_feat"] = img_features_hdf["features"][img_feature_index]
        assert image_id == img_features_hdf["image_id"][img_feature_index]

    return features

  def keys(self) -> List[int]:
    return self.text_feature_id_l

  @property
  def split(self):
    return self._split

  def get_old_img_ids(self, visdial_jsonpath):
    with open(visdial_jsonpath, "r") as visdial_file:
      visdial_data = json.load(visdial_file)

    return [dialog_for_image["image_id"] for dialog_for_image in visdial_data["data"]["dialogs"]]

  def get_grid_bboxes(self, img_w, img_h, size=7):
    grid_w = img_w / size
    grid_h = img_h / size
    w_points, h_points = [], []
    a, b = 0, 0
    for i in range(size):
      w_points.append(a)
      h_points.append(b)
      a += grid_w
      b += grid_h
    grid_bboxes = np.zeros((size, size, 4))
    for y_idx, grid_y in enumerate(h_points):
      for x_idx, grid_x in enumerate(w_points):
        grid_bboxes[x_idx, y_idx, 0] = grid_x
        grid_bboxes[x_idx, y_idx, 1] = grid_y
        grid_bboxes[x_idx, y_idx, 2] = grid_x + grid_w
        grid_bboxes[x_idx, y_idx, 3] = grid_y + grid_h
    return grid_bboxes.reshape(-1, 4)
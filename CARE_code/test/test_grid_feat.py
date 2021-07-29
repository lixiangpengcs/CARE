import h5py
import os
import numpy as np

root_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/grid_feat/'

with h5py.File(os.path.join(root_path, 'grid_feature_X101_train.hdf5'), 'r') as grid_feature:
    grid_features = np.array(grid_feature['grid_features'])
    print(grid_features.shape)
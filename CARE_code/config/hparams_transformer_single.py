import random
from collections import defaultdict

OLD_DATASET_PARAMS = defaultdict(
  dataset_version='0.9',
  visdial_json='/raid/lixiangpeng/dataset/visual_dialog/data/v0.9/visdial_0.9_%s.json',
)

TransBASE_PARAMS = defaultdict(
  # Dataset reader arguments
  dataset_version='1.0',

  img_feature_type="dan_faster_rcnn_x101", # faster_rcnn_x101, dan_faster_rcnn_x101
  model_train_type="single", # single, multi
  img_features_h5='/raid/lixiangpeng/dataset/visual_dialog/data/visdial_1.0_img/features_%s_%s.h5', # img_feature_type | train, val, test
  imgid2idx_path='/raid/lixiangpeng/dataset/visual_dialog/data/visdial_1.0_img/%s_imgid2idx.pkl', # dan_img - train, val, test
  text_features_h5='/raid/lixiangpeng/dataset/visual_dialog/data/visdial_1.0_text/visdial_1.0_%s_text_%s.hdf5',

  word_counts_json='/raid/lixiangpeng/dataset/visual_dialog/data/v1.0/visdial_1.0_word_counts_train.json',
  glove_npy='/raid/lixiangpeng/dataset/visual_dialog/data/visdial_1.0_text/glove.npy',
  pretrained_glove='/raid/lixiangpeng/dataset/visual_dialog/data/word_embeddings/glove/glove.6B.300d.txt',

  visdial_json='/raid/lixiangpeng/dataset/visual_dialog/data/v1.0/visdial_1.0_%s.json',
  valid_dense_json='/raid/lixiangpeng/dataset/visual_dialog/data/v1.0/visdial_1.0_val_dense_annotations.json',

  # Model save arguments
  root_dir='/home/lixiangpeng/data/models/visdial/', # for saving logs, checkpoints and result files
  save_dirpath ='checkpoints/',
  load_pthpath='',

  # grid feature trigger
  grid_feat_trigger = True,
  grid_feature_h5 = '/home/lixiangpeng/data/dataset/visual_dialog/data/grid_feat/grid_feature_X101_%s.hdf5',

  grid_region_trigger = True,
  grid_region_path = '/home/lixiangpeng/data/dataset/visual_dialog/data/grid_feat/region_feats/visdial/%s',

  img_norm=True,
  max_sequence_length=20,
  vocab_min_count=5,


  # test_trigger
  test_trigger=True,

  # Train related arguments
  gpu_ids=[0,1],
  cpu_workers=4,
  tensorboard_step=100,
  do_vaild=True,
  overfit=False,
  # random_seed=random.sample(range(1000, 10000), 1), # 3143
  random_seed=[3143],
  concat_history=True,

  # currculum trigger
  curriculum_trigger=False,
  imgid2clidx_path='/home/lixiangpeng/data/dataset/visual_dialog/data/CL_data/train_imgid2clidx.pkl',
  cl_data_path='/home/lixiangpeng/data/dataset/visual_dialog/data/CL_data/train_simordermat.npy',
  imgid2weight_path='/home/lixiangpeng/data/dataset/visual_dialog/data/CL_data/train_imgid2cfdweight_iq.pkl',
  loss_type='cl_negative_with_cl_positive',
  # fixed_negative_loss, only_fixed_negative_loss, cl_negative_loss, cl_negative_with_fixed_positive, cl_negative_with_cl_positive
  # inter_cl, inter_cl_intra_cl, inter_spl

  loss_margin=0.05,
  pos_sample_number=3,
  neg_sample_number=10,
  neg_lower_bound=10,
  pos_upper_bound=5, # it can be decreased into 3 in future
  loss_lambda=5.0,

  # Opitimization related arguments
  num_epochs=15,
  train_batch_size=32, #32 x num_gpus is a good rule of thumb
  eval_batch_size=32,
  virtual_batch_size=32,
  training_splits="train",
  evaluation_type="disc",
  lr_scheduler="LambdaLR",
  warmup_epochs=2,
  warmup_factor=0.01,
  initial_lr=0.0002,
  lr_gamma=0.1,
  lr_milestones=[5],  # epochs when lr —> lr * lr_gamma
  lr_gamma2=0.1,
  lr_milestones2=[7],  # epochs when lr —> lr * lr_gamma2

  # Model related arguments
  encoder='transformermodel', # [basemodel, mvan, transformermodel]
  decoder='disc',  # [disc,gen]

  # for transformer model
  qih_bert_layers=3,

  img_feature_size=2048,
  word_embedding_size=300,
  lstm_hidden_size=512,
  hidden_size=512,
  lstm_num_layers=2,
  dropout=0.4,
  dropout_fc=0.25,
)

Trans_MULTI_PARAMS= TransBASE_PARAMS.copy()
Trans_MULTI_PARAMS.update(
  gpu_ids=[0],
  num_epochs=15,
  train_batch_size=32, # 32 x num_gpus is a good rule of thumb
  eval_batch_size=32,
  virtual_batch_size=32,
  cpu_workers=4,

  # Model related arguments
  encoder='mvan',
  decoder='disc_gen',  # [disc,gen]
  evaluation_type="disc_gen",
  aggregation_type="average",
)
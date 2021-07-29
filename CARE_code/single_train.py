import os
import logging
import itertools

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from tqdm import tqdm
from setproctitle import setproctitle
from bisect import bisect

from datetime import datetime
import numpy as np

from data.dataset import VisDialDataset
from visdial.encoders import Encoder
from visdial.decoders import Decoder
from visdial.model import EncoderDecoderModel
from visdial.utils.checkpointing import CheckpointManager, load_checkpoint

from single_evaluation import Evaluation
# from torch.cuda.amp import autocast as autocast
# from torch.cuda.amp import GradScaler as GradScaler

class MVAN(object):
  def __init__(self, hparams):
    self.hparams = hparams
    self._logger = logging.getLogger(__name__)

    np.random.seed(hparams.random_seed[0])
    torch.manual_seed(hparams.random_seed[0])
    torch.cuda.manual_seed_all(hparams.random_seed[0])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    self.device = (
      torch.device("cuda")
    )

    setproctitle(hparams.dataset_version + '_' + hparams.model_name + '_' + str(hparams.random_seed[0]))

  # def _build_data_process(self):
  def _build_dataloader(self):
    # =============================================================================
    #   SETUP DATASET, DATALOADER
    # =============================================================================
    old_split = "train" if self.hparams.dataset_version == "0.9" else None
    self.train_dataset = VisDialDataset(
      self.hparams,
      overfit=self.hparams.overfit,
      split="train",
      old_split = old_split
    )

    collate_fn = None
    if "dan" in self.hparams.img_feature_type:
      collate_fn = self.train_dataset.collate_fn

    self.train_dataloader = DataLoader(
      self.train_dataset,
      batch_size=self.hparams.train_batch_size,
      num_workers=self.hparams.cpu_workers,
      shuffle=True,
      drop_last=True,
      collate_fn=collate_fn,
    )

    print("""
      # -------------------------------------------------------------------------
      #   DATALOADER FINISHED
      # -------------------------------------------------------------------------
      """)

  def _build_model(self):

    # =============================================================================
    #   MODEL : Encoder, Decoder
    # =============================================================================

    print('\t* Building model...')
    # Pass vocabulary to construct Embedding layer.
    encoder = Encoder(self.hparams, self.train_dataset.vocabulary)
    decoder = Decoder(self.hparams, self.train_dataset.vocabulary)

    print("Encoder: {}".format(self.hparams.encoder))
    print("Decoder: {}".format(self.hparams.decoder))

    # New: Initializing word_embed using GloVe
    if self.hparams.glove_npy != '':
      encoder.word_embed.weight.data = torch.from_numpy(np.load(self.hparams.glove_npy))
      print("Loaded glove vectors from {}".format(self.hparams.glove_npy))
    # Share word embedding between encoder and decoder.
    decoder.word_embed = encoder.word_embed

    # Wrap encoder and decoder in a model.
    self.model = EncoderDecoderModel(encoder, decoder)
    self.model = self.model.to(self.device)
    # Use Multi-GPUs
    if -1 not in self.hparams.gpu_ids and len(self.hparams.gpu_ids) > 1:
      self.model = nn.DataParallel(self.model, self.hparams.gpu_ids)

    # =============================================================================
    #   CRITERION
    # =============================================================================
    if "disc" in self.hparams.decoder:
      self.criterion = nn.CrossEntropyLoss(reduction='none')

    elif "gen" in self.hparams.decoder:
      self.criterion = nn.CrossEntropyLoss(ignore_index=self.train_dataset.vocabulary.PAD_INDEX)

    # Total Iterations -> for learning rate scheduler
    if self.hparams.training_splits == "trainval":
      self.iterations = (len(self.train_dataset) + len(self.valid_dataset)) // self.hparams.virtual_batch_size
    else:
      self.iterations = len(self.train_dataset) // self.hparams.virtual_batch_size

    # =============================================================================
    #   OPTIMIZER, SCHEDULER
    # =============================================================================

    def lr_lambda_fun(current_iteration: int) -> float:
      """Returns a learning rate multiplier.

      Till `warmup_epochs`, learning rate linearly increases to `initial_lr`,
      and then gets multiplied by `lr_gamma` every time a milestone is crossed.
      """
      current_epoch = float(current_iteration) / self.iterations
      if current_epoch <= self.hparams.warmup_epochs:
        alpha = current_epoch / float(self.hparams.warmup_epochs)
        return self.hparams.warmup_factor * (1.0 - alpha) + alpha
      else:
        return_val = 1.0
        if current_epoch >= self.hparams.lr_milestones[0] and current_epoch < self.hparams.lr_milestones2[0]:
          idx = bisect(self.hparams.lr_milestones, current_epoch)
          return_val = pow(self.hparams.lr_gamma, idx)
        elif current_epoch >= self.hparams.lr_milestones2[0]:
          idx = bisect(self.hparams.lr_milestones2, current_epoch)
          return_val = self.hparams.lr_gamma * pow(self.hparams.lr_gamma2, idx)
        return return_val

    if self.hparams.lr_scheduler == "LambdaLR":
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.hparams.initial_lr)
      self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda_fun)
    else:
      raise NotImplementedError

    print(
      """
      # -------------------------------------------------------------------------
      #   Model Build Finished
      # -------------------------------------------------------------------------
      """
    )

  def _setup_training(self):
    if self.hparams.save_dirpath == 'checkpoints/':
      self.save_dirpath = os.path.join(self.hparams.root_dir, self.hparams.save_dirpath)
    self.summary_writer = SummaryWriter(self.save_dirpath)
    self.checkpoint_manager = CheckpointManager(
      self.model, self.optimizer, self.save_dirpath, hparams=self.hparams
    )

    # If loading from checkpoint, adjust start epoch and load parameters.
    if self.hparams.load_pthpath == "":
      self.start_epoch = 1
    else:
      # "path/to/checkpoint_xx.pth" -> xx
      self.start_epoch = int(self.hparams.load_pthpath.split("_")[-1][:-4])
      self.start_epoch += 1
      model_state_dict, optimizer_state_dict = load_checkpoint(self.hparams.load_pthpath)
      if isinstance(self.model, nn.DataParallel):
        self.model.module.load_state_dict(model_state_dict)
      else:
        self.model.load_state_dict(model_state_dict)
      self.optimizer.load_state_dict(optimizer_state_dict)
      self.previous_model_path = self.hparams.load_pthpath
      print("Loaded model from {}".format(self.hparams.load_pthpath))

    print(
      """
      # -------------------------------------------------------------------------
      #   Setup Training Finished
      # -------------------------------------------------------------------------
      """
    )

  def _spl_setting(self):
    self.v = torch.zeros(len(self.train_dataset)).float().to(self.device)
    self.cnt = 0
    self.threshold = 3.0
    self.growing_factor = 1.10

  def _loss_fn(self, epoch, batch, output):
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))
    return batch_loss

  def _fixed_negative_loss(self, batch_iter, epoch, batch, output, margin=0.1, neg_sample_number=10):
    batch_size = output.size(0)
    targets = batch['ans_ind'].view(-1).cuda()
    output = output.view(-1, 100)
    scores = F.softmax(output, dim=-1)

    negative_idxs = torch.randint(self.hparams.neg_lower_bound, 100, (batch_size, 10, neg_sample_number)).cuda()
    negative_idxs_view = negative_idxs.view(-1, neg_sample_number)

    sim_order = batch['cl_order'].cuda() # bs x 10 x 100
    sim_order_view = sim_order.view(-1, 100)
    gt_scores = torch.gather(scores, dim=-1, index=targets.unsqueeze(1))
    negative_labels = torch.gather(sim_order_view, dim=-1, index=negative_idxs_view)
    negative_scores = torch.gather(scores, dim=-1, index=negative_labels)
    margin_loss = torch.nn.functional.relu(margin-gt_scores+negative_scores)
    loss = torch.mean(margin_loss, dim=-1)

    return torch.mean(loss)

  def _cl_negative_loss(self, batch_iter, epoch, batch, output, margin=0.1, neg_sample_number=10):
    batch_size = output.size(0)
    targets = batch['ans_ind'].view(-1).cuda()
    output = output.view(-1, 100)
    scores = F.softmax(output, dim=-1)

    slope = -(100. - 20.) / (self.iterations * 10)
    upper_bound = max(slope * batch_iter + 100, 20)
    negative_idxs = torch.randint(self.hparams.neg_lower_bound, round(upper_bound), (batch_size, 10, neg_sample_number)).cuda()
    negative_idxs_view = negative_idxs.view(-1, neg_sample_number)

    sim_order = batch['cl_order'].cuda() # bs x 10 x 100
    sim_order_view = sim_order.view(-1, 100)
    gt_scores = torch.gather(scores, dim=-1, index=targets.unsqueeze(1))
    negative_labels = torch.gather(sim_order_view, dim=-1, index=negative_idxs_view)
    negative_scores = torch.gather(scores, dim=-1, index=negative_labels)
    margin_loss = torch.nn.functional.relu(margin-gt_scores+negative_scores)
    loss = torch.mean(margin_loss, dim=-1)
    if batch_iter % 1000 == 0:
      print('Current sampling scope: [%d, %d]'%(10, round(upper_bound)))
    return torch.mean(loss)

  def _cl_negative_with_fixed_positive_loss(self, batch_iter, epoch, batch, output, margin=0.1, pos_sample_number=5, neg_sample_number=10):
    batch_size = output.size(0)
    output = output.view(-1, 100)
    scores = F.softmax(output, dim=-1)

    positive_idxs = torch.arange(0, pos_sample_number).cuda()
    positive_idxs = positive_idxs.unsqueeze(0).unsqueeze(1)
    positive_idxs = positive_idxs.repeat(batch_size, 10, 1)
    positive_idxs_view = positive_idxs.view(-1, pos_sample_number)
    slope = -(100. - 20.) / (self.iterations * 10)
    negative_upper_bound = max(slope * batch_iter + 100, 20)
    negative_idxs = torch.randint(self.hparams.neg_lower_bound, round(negative_upper_bound), (batch_size, 10, neg_sample_number)).cuda()
    negative_idxs_view = negative_idxs.view(-1, neg_sample_number)

    sim_order = batch['cl_order'].to(self.device)  # bs x 10 x 100
    sim_order_view = sim_order.view(-1, 100)

    positive_labels = torch.gather(sim_order_view, dim=-1, index=positive_idxs_view)
    positive_scores = torch.gather(scores, dim=-1, index=positive_labels)
    pos_scores_unsq = positive_scores.unsqueeze(1)
    negative_labels = torch.gather(sim_order_view, dim=-1, index=negative_idxs_view)
    negative_scores = torch.gather(scores, dim=-1, index=negative_labels)
    neg_scores_unsq = negative_scores.unsqueeze(2)
    margin_loss = torch.nn.functional.relu(margin - pos_scores_unsq + neg_scores_unsq)
    loss = torch.mean(margin_loss, dim=[1, 2])
    if batch_iter % 1000 == 0:
      print('Current negative sampling scope: [%d, %d]' % (self.hparams.neg_lower_bound, round(negative_upper_bound)))
    return torch.mean(loss)

  def _cl_negative_with_cl_positive_loss(self, batch_iter, epoch, batch, output, margin=0.1, pos_sample_number=3, neg_sample_number=10):
    batch_size = output.size(0)
    output = output.view(-1, 100)
    scores = F.softmax(output, dim=-1)

    #positive_upper_bound = max(epoch, self.hparams.pos_upper_bound) # (epoch, 3)
    positive_upper_bound = max(12-epoch, self.hparams.pos_upper_bound) #

    positive_idxs = torch.randint(0, positive_upper_bound, (batch_size, 10, pos_sample_number)).to(self.device)
    positive_idxs_view = positive_idxs.view(-1, pos_sample_number)

    slope = -(100. - 20.) / (self.iterations * 10)
    negative_upper_bound = max(slope * batch_iter + 100, 20)
    negative_idxs = torch.randint(self.hparams.neg_lower_bound, round(negative_upper_bound), (batch_size, 10, neg_sample_number)).to(self.device)
    negative_idxs_view = negative_idxs.view(-1, neg_sample_number)

    sim_order = batch['cl_order'].to(self.device) # bs x 10 x 100
    sim_order_view = sim_order.view(-1, 100)

    positive_labels = torch.gather(sim_order_view, dim=-1, index=positive_idxs_view)
    positive_scores = torch.gather(scores, dim=-1, index=positive_labels)
    pos_scores_unsq = positive_scores.unsqueeze(1)
    negative_labels = torch.gather(sim_order_view, dim=-1, index=negative_idxs_view)
    negative_scores = torch.gather(scores, dim=-1, index=negative_labels)
    neg_scores_unsq = negative_scores.unsqueeze(2)
    margin_loss = torch.nn.functional.relu(margin - pos_scores_unsq + neg_scores_unsq)
    loss = torch.mean(margin_loss, dim=[1, 2])
    if batch_iter % 1000 == 0:
      print('Current negative sampling scope: [%d, %d], Current negative sampling scope: [%d, %d]' % (0, positive_upper_bound, self.hparams.neg_lower_bound, round(negative_upper_bound)))
    return torch.mean(loss)

  def _inter_cl_margin_loss(self, batch_iter, epoch, batch, output, margin=0.1, pos_sample_number=5, neg_sample_number=10):
    start_threshold = 0.4
    start_iteration = 2000
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))
    batch_size = target.size(0)
    batch_loss = torch.mean(batch_loss.view(batch_size, -1), dim=-1)  # bs x 1
    sample_weights = batch['sample_weight'].cuda()
    stop_threshold = self.iterations*10
    slope = (1. - start_threshold) / (stop_threshold - start_iteration)

    if batch_iter <= start_iteration:
      threshold = torch.tensor(start_threshold).cuda()
    elif batch_iter < stop_threshold:
      threshold = torch.tensor(slope * (batch_iter - start_iteration) + start_threshold).cuda()
    else:
      threshold = torch.tensor(1.).cuda()
    sampled_weights = (sample_weights < threshold).int()
    if batch_iter % 1000 == 0:
      print("current threshold: ", threshold.item(), torch.sum(sample_weights).item())
    if torch.eq(torch.sum(sampled_weights), 0).item() == True:
      batch_loss = torch.sum(batch_loss * sampled_weights) * 0.0   # here is not the question
    else:
      batch_loss = torch.sum(batch_loss * sampled_weights) / torch.sum(sampled_weights)
    return batch_loss

  def _inter_sqrtcl_margin_loss(self, batch_iter, epoch, batch, output, margin=0.1, pos_sample_number=5, neg_sample_number=10):
    start_threshold = 0.6
    start_iteration = 3800
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))
    batch_size = target.size(0)
    batch_loss = torch.mean(batch_loss.view(batch_size, -1), dim=-1)  # bs x 1
    sample_weights = batch['sample_weight'].cuda()
    stop_threshold = self.iterations * 10
    slope = (1.0 - start_threshold*start_threshold) / (stop_threshold-start_iteration)

    if batch_iter <= start_iteration:
      threshold = torch.tensor(start_threshold).cuda()
    else:
      threshold = torch.tensor(min((slope * (batch_iter - start_iteration) + start_threshold*start_threshold)**0.5, 1)).cuda()
    sampled_weights = (sample_weights<threshold).int()
    batch_loss = torch.sum(batch_loss * sampled_weights) / torch.sum(sampled_weights)
    return batch_loss


  def _inter_softspl_margin_loss(self, batch_iter, epoch, batch, output):
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))
    batch_size = target.size(0)
    batch_loss = torch.mean(batch_loss.view(batch_size, -1), dim=-1)  # bs x 1
    index = batch['index']
    discount = 0.5
    if self.cnt < 1100:
      if self.cnt==0:
        print('-----Warm up is starting-----')
      loss = torch.mean(batch_loss)
    else:
      if self.cnt==1100:
        print('--------------warming up is done, SPL is adopted -----------')
      v_flag = batch_loss < self.threshold   # attain easy samples indexes
      v = (v_flag * (1 - batch_loss / self.threshold)).float()   # update weight for each
      self.v[index] = v
      f_losses = batch_loss * v   # calculate new loss for current stage
      f_loss = torch.mean(f_losses) # mean loss
      reg = discount * self.threshold * torch.mean(v * v - 2 * v)   # calculate SPL regularizer
      loss = f_loss + reg
    self.cnt += 1
    if self.cnt%self.iterations == 0:
      if epoch<8:
        self.threshold *= self.growing_factor
        print(torch.sum(self.v).item(), "Current Threshold: ", self.threshold)
        # print('SPL consists of SPL loss: %5.4d, and reg loss: %5.4d'%(f_loss.item(), reg.item()))
      else:
        v_sum = 1
    return loss

  def _inter_cl_intra_cl_margin_loss(self, batch_iter, epoch, batch, output, margin=0.1, pos_sample_number=5, neg_sample_number=10):
    #
    target = (batch["ans_ind"] if "disc" in self.hparams.decoder else batch["ans_out"])
    batch_loss = self.criterion(output.view(-1, output.size(-1)), target.view(-1).to(self.device))
    batch_size = target.size(0)
    batch_loss = torch.mean(batch_loss.view(batch_size, -1), dim=-1)  # bs x 1
    sample_weights = batch['sample_weight'].cuda()
    slope = (1. - 0.2) / (self.iterations * 10. - 2000)
    if batch_iter <= 2000:
      threshold = torch.tensor(0.2).cuda()
    elif batch_iter < self.iterations*10:
      threshold = torch.tensor(slope * (batch_iter - 2000) + 0.2).cuda()
    else:
      threshold = torch.tensor(1.).cuda()
    sampled_weight_flags = (sample_weights < threshold).int()
    if batch_iter % 1000 == 0:
      print("current threshold: ", threshold.item())
    if torch.eq(torch.sum(sampled_weight_flags), 0).item() == True:
      ce_batch_loss = torch.sum(batch_loss * sampled_weight_flags)
    else:
      ce_batch_loss = torch.sum(batch_loss * sampled_weight_flags) / torch.sum(sampled_weight_flags)

    output = output.view(-1, 100)
    scores = F.softmax(output, dim=-1)
    positive_upper_bound = max(epoch, 3)

    positive_idxs = torch.randint(0, positive_upper_bound, (batch_size, 10, pos_sample_number)).cuda()
    positive_idxs_view = positive_idxs.view(-1, pos_sample_number)
    slope = -(100. - 20.) / (self.iterations * 10)
    negative_upper_bound = max(slope * batch_iter + 100, 20)
    negative_idxs = torch.randint(10, round(negative_upper_bound), (batch_size, 10, neg_sample_number)).cuda()
    negative_idxs_view = negative_idxs.view(-1, neg_sample_number)
    sim_order = batch['cl_order'].cuda()  # bs x 10 x 100
    sim_order_view = sim_order.view(-1, 100)

    positive_labels = torch.gather(sim_order_view, dim=-1, index=positive_idxs_view)
    positive_scores = torch.gather(scores, dim=-1, index=positive_labels)
    pos_scores_unsq = positive_scores.unsqueeze(1)
    negative_labels = torch.gather(sim_order_view, dim=-1, index=negative_idxs_view)
    negative_scores = torch.gather(scores, dim=-1, index=negative_labels)
    neg_scores_unsq = negative_scores.unsqueeze(2)
    margin_loss = torch.nn.functional.relu(margin - pos_scores_unsq + neg_scores_unsq)
    loss = torch.mean(margin_loss, dim=[1, 2])
    if torch.eq(torch.sum(sampled_weight_flags), 0).item() == True:
      cl_batch_loss = torch.sum(loss * sampled_weight_flags)
    else:
      cl_batch_loss = torch.sum(loss * sampled_weight_flags) / torch.sum(sampled_weight_flags)
    if batch_iter % 1000 == 0:
      print('Current negative sampling scope: [%d, %d]' % (10, round(negative_upper_bound)))
    final_loss = ce_batch_loss + cl_batch_loss
    return final_loss

  def train(self):

    self._build_dataloader()
    self._build_model()
    self._setup_training()
    self._spl_setting()
    # scaler = GradScaler()
    # Evaluation Setup
    evaluation = Evaluation(self.hparams, model=self.model, split="val")

    # Forever increasing counter to keep track of iterations (for tensorboard log).
    global_iteration_step = (self.start_epoch - 1) * self.iterations

    running_loss = 0.0  # New
    train_begin = datetime.utcnow()  # New
    batch_iter = 0
    print(
      """
      # -------------------------------------------------------------------------
      #   Model Train Starts (NEW)
      # -------------------------------------------------------------------------
      """
    )
    for epoch in range(self.start_epoch, self.hparams.num_epochs):
      self.model.train()
      # -------------------------------------------------------------------------
      #   ON EPOCH START  (combine dataloaders if training on train + val)
      # -------------------------------------------------------------------------
      combined_dataloader = itertools.chain(self.train_dataloader)

      print(f"\nTraining for epoch {epoch}:", "Total Iter:", self.iterations)
      tqdm_batch_iterator = tqdm(combined_dataloader)
      accumulate_batch = 0 # taesun New

      for i, batch in enumerate(tqdm_batch_iterator):
        buffer_batch = batch.copy()
        for key in batch:
          buffer_batch[key] = buffer_batch[key].to(self.device)

        output = self.model(buffer_batch)
        batch_loss = 0
        if not self.hparams.curriculum_trigger:
          batch_loss = batch_loss + torch.mean(self._loss_fn(epoch, batch, output))
        elif self.hparams.loss_type == 'fixed_negative_loss':
          ce_loss = torch.mean(self._loss_fn(epoch, batch, output))
          sample_loss = self._fixed_negative_loss(batch_iter, epoch, batch, output, margin=self.hparams.loss_margin,
                                                  neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + ce_loss + self.hparams.loss_lambda*sample_loss

        elif self.hparams.loss_type == 'only_fixed_negative_loss':
          # ce_loss = torch.mean(self._loss_fn(epoch, batch, output))
          sample_loss = self._fixed_negative_loss(batch_iter, epoch, batch, output, margin=self.hparams.loss_margin,
                                                  neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + sample_loss

        elif self.hparams.loss_type == 'cl_negative_loss':
          # print('cl loss computation')
          ce_loss = torch.mean(self._loss_fn(epoch, batch, output))
          cl_loss = self._cl_negative_loss(batch_iter, epoch, batch, output, margin=self.hparams.loss_margin,
                                           neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + ce_loss + self.hparams.loss_lambda*cl_loss
          if batch_iter % 1000 == 0:
            print('Current loss : [%5.4f, %5.4f]' % (ce_loss, cl_loss))
        elif self.hparams.loss_type == 'cl_negative_with_fixed_positive':
          # print('cl loss computation')
          ce_loss = torch.mean(self._loss_fn(epoch, batch, output))
          cl_loss = self._cl_negative_with_fixed_positive_loss(batch_iter, epoch, batch, output,
                                                                   margin=self.hparams.loss_margin,
                                                          pos_sample_number=self.hparams.pos_sample_number,
                                                          neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + ce_loss + self.hparams.loss_lambda*cl_loss
          if batch_iter % 1000 == 0:
            print('Current loss : [%5.4f, %5.4f]' % (ce_loss, cl_loss))

        elif self.hparams.loss_type == 'cl_negative_with_cl_positive':
          # print('cl loss computation')
          ce_loss = torch.mean(self._loss_fn(epoch, batch, output))
          cl_loss = self._cl_negative_with_cl_positive_loss(batch_iter, epoch, batch, output,
                                                                margin=self.hparams.loss_margin,
                                                       pos_sample_number=self.hparams.pos_sample_number,
                                                       neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + ce_loss + self.hparams.loss_lambda*cl_loss
          if batch_iter % 1000 == 0:
            print('Current loss : [%5.4f, %5.4f]' % (ce_loss, cl_loss))

        elif self.hparams.loss_type == 'inter_cl':
          # print('cl loss computation')
          cl_loss = self._inter_cl_margin_loss(batch_iter, epoch, batch, output, margin=self.hparams.loss_margin,
                                               pos_sample_number=self.hparams.pos_sample_number,
                                               neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + cl_loss
          if batch_iter % 1000 == 0:
            print('Current loss : [%5.4f]' % (cl_loss))

        elif self.hparams.loss_type == 'inter_sqrtcl':
          cl_loss = self._inter_sqrtcl_margin_loss(batch_iter, epoch, batch, output, margin=self.hparams.loss_margin,
                                               pos_sample_number=self.hparams.pos_sample_number,
                                               neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + cl_loss
          if batch_iter % 1000 == 0:
            print('Current loss : [%5.4f]' % (cl_loss))

        elif self.hparams.loss_type == 'inter_cl_intra_cl':
          # print('cl loss computation')
          cl_loss = self._inter_cl_intra_cl_margin_loss(batch_iter, epoch, batch, output, margin=self.hparams.loss_margin,
                                                        pos_sample_number=self.hparams.pos_sample_number,
                                                        neg_sample_number=self.hparams.neg_sample_number)
          batch_loss = batch_loss + cl_loss
          if batch_iter % 1000 == 0:
            print('Current loss : [%5.4f]' % (cl_loss))
        elif self.hparams.loss_type == 'inter_spl':
          spl_loss = self._inter_softspl_margin_loss(batch_iter, epoch, batch, output)
          batch_loss = batch_loss + spl_loss

        batch_loss.backward()
        batch_iter += 1
        accumulate_batch += batch["img_ids"].shape[0]
        if self.hparams.virtual_batch_size == accumulate_batch \
            or i == (len(self.train_dataset) // self.hparams.train_batch_size): # last batch

          self.optimizer.step()

          # --------------------------------------------------------------------
          #    Update running loss and decay learning rates
          # --------------------------------------------------------------------
          if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
          else:
            running_loss = batch_loss.item()

          self.optimizer.zero_grad()
          accumulate_batch = 0

          self.scheduler.step(global_iteration_step)

          global_iteration_step += 1
          # torch.cuda.empty_cache()
          description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
            datetime.utcnow() - train_begin,
            epoch,
            global_iteration_step, running_loss,
            self.optimizer.param_groups[0]['lr'])
          tqdm_batch_iterator.set_description(description)

          # tensorboard
          if global_iteration_step % self.hparams.tensorboard_step == 0:
            description = "[{}][Epoch: {:3d}][Iter: {:6d}][Loss: {:6f}][lr: {:7f}]".format(
              datetime.utcnow() - train_begin,
              epoch,
              global_iteration_step, running_loss,
              self.optimizer.param_groups[0]['lr'],
              )
            self._logger.info(description)
            # tensorboard writing scalar
            self.summary_writer.add_scalar(
              "train/loss", batch_loss, global_iteration_step
            )
            self.summary_writer.add_scalar(
              "train/lr", self.optimizer.param_groups[0]["lr"], global_iteration_step
            )

      # -------------------------------------------------------------------------
      #   ON EPOCH END  (checkpointing and validation)
      # -------------------------------------------------------------------------
      self.checkpoint_manager.step(epoch)
      self.previous_model_path = os.path.join(self.checkpoint_manager.ckpt_dirpath, "checkpoint_%d.pth" % (epoch))
      self._logger.info(self.previous_model_path)

      if epoch < self.hparams.num_epochs - 1 and self.hparams.dataset_version == '0.9':
        continue

      torch.cuda.empty_cache()
      evaluation.run_evaluate(self.previous_model_path, global_iteration_step, self.summary_writer,
                              os.path.join(self.checkpoint_manager.ckpt_dirpath, "ranks_%d_valid.json" % epoch))
      torch.cuda.empty_cache()

    return self.previous_model_path
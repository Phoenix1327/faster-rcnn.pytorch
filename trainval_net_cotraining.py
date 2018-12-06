# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import random
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='vgg16', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="models",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=0, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.001, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
  elif args.dataset == "vg":
      # train sizes: train, smalltrain, minitrain
      # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = True
  cfg.USE_GPU_NMS = args.cuda
  cfg.TRAIN.PER_SUP = 0.5

  pdb.set_trace()
  imdb, ori_roidb, ori_ratio_list, ori_ratio_index = combined_roidb(args.imdb_name)
  print('{:d} roidb entries'.format(len(ori_roidb)))

  train_size = len(ori_roidb)
  ori_roidb_idx = np.arange(train_size)
  if cfg.TRAIN.USE_FLIPPED:
      roidb_idx = np.arange(int(train_size/2.0))
      sampled_roidb_idx = np.random.choice(roidb_idx, size=int(train_size*0.5*cfg.TRAIN.PER_SUP), replace=False)
      flipped_sampled_roidb_idx = sampled_roidb_idx + int(train_size/2.0)
      sampled_roidb_idx = np.hstack((sampled_roidb_idx, flipped_sampled_roidb_idx))
  else:
      roidb_idx = np.arange(train_size)
      sampled_roidb_idx = np.random.choice(roidb_idx, size=train_size, replace=False)

  unsup_roidb_idx = np.setdiff1d(ori_roidb_idx, sampled_roidb_idx, assume_unique=True)


  #roidb = map(ori_roidb.__getitem__, np.sort(sampled_roidb_idx))
  #pdb.set_trace()
  sampled_ratio_list = [ori_ratio_list[idx] for idx, val in enumerate(ori_ratio_index) if np.isin(val, sampled_roidb_idx)]
  #pdb.set_trace()
  ratio_list = np.asarray(sampled_ratio_list)
  sampled_ratio_index = [val for val in ori_ratio_index if np.isin(val, sampled_roidb_idx)]
  #pdb.set_trace()
  ratio_index = np.asarray(sampled_ratio_index)

  train_size = ratio_list.shape[0]
  print('sample semi-supervised {:d} roidb entries. Done!'.format(train_size))

  
  #unsup_roidb = map(ori_roidb.__getitem__, np.sort(unsup_roidb_idx))
  unsup_ratio_list = [ori_ratio_list[idx] for idx, val in enumerate(ori_ratio_index) if np.isin(val, unsup_roidb_idx)]
  unsup_ratio_list = np.asarray(unsup_ratio_list)
  unsup_ratio_index = [val for val in ori_ratio_index if np.isin(val, unsup_roidb_idx)]
  unsup_ratio_index = np.asarray(unsup_ratio_index)

  unsup_train_size = unsup_ratio_list.shape[0]
  print('Remaining unsupervised {:d} roidb entries. Done!'.format(unsup_train_size))

  #pdb.set_trace()

  output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # two supervised datum streams
  sup_sampler_batch_1 = sampler(train_size, args.batch_size)
  sup_sampler_batch_2 = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(ori_roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True)

  sup_dataloader_1 = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sup_sampler_batch_1, num_workers=args.num_workers)
  
  sup_dataloader_2 = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=sup_sampler_batch_2, num_workers=args.num_workers)

  # one unsupervised datum stream
  unsup_sampler_batch = sampler(unsup_train_size, args.batch_size)
  unsup_dataset = roibatchLoader(ori_roidb, unsup_ratio_list, unsup_ratio_index, \
                           args.batch_size, imdb.num_classes, training=True)
  unsup_dataloader = torch.utils.data.DataLoader(unsup_dataset, batch_size=args.batch_size, sampler=unsup_sampler_batch, num_workers=args.num_workers)

  #pdb.set_trace()



  # initilize the tensor holder here.
  sup_im_data_1 = torch.FloatTensor(1)
  sup_im_info_1 = torch.FloatTensor(1)
  sup_num_boxes_1 = torch.LongTensor(1)
  sup_gt_boxes_1 = torch.FloatTensor(1)
  sup_im_data_2 = torch.FloatTensor(1)
  sup_im_info_2 = torch.FloatTensor(1)
  sup_num_boxes_2 = torch.LongTensor(1)
  sup_gt_boxes_2 = torch.FloatTensor(1)
  unsup_im_data = torch.FloatTensor(1)
  unsup_im_info = torch.FloatTensor(1)
  unsup_num_boxes = torch.LongTensor(1)
  unsup_gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    sup_im_data_1 = sup_im_data_1.cuda()
    sup_im_info_1 = sup_im_info_1.cuda()
    sup_num_boxes_1 = sup_num_boxes_1.cuda()
    sup_gt_boxes_1 = sup_gt_boxes_1.cuda()
    sup_im_data_2 = sup_im_data_2.cuda()
    sup_im_info_2 = sup_im_info_2.cuda()
    sup_num_boxes_2 = sup_num_boxes_2.cuda()
    sup_gt_boxes_2 = sup_gt_boxes_2.cuda()
    unsup_im_data = unsup_im_data.cuda()
    unsup_im_info = unsup_im_info.cuda()
    unsup_num_boxes = unsup_num_boxes.cuda()
    unsup_gt_boxes = unsup_gt_boxes.cuda()

  # make variable
  sup_im_data_1 = Variable(sup_im_data_1, requires_grad=True)
  sup_im_info_1 = Variable(sup_im_info_1)
  sup_num_boxes_1 = Variable(sup_num_boxes_1)
  sup_gt_boxes_1 = Variable(sup_gt_boxes_1)
  sup_im_data_2 = Variable(sup_im_data_2, requires_grad=True)
  sup_im_info_2 = Variable(sup_im_info_2)
  sup_num_boxes_2 = Variable(sup_num_boxes_2)
  sup_gt_boxes_2 = Variable(sup_gt_boxes_2)
  unsup_im_data = Variable(unsup_im_data, requires_grad=True)
  unsup_im_info = Variable(unsup_im_info)
  unsup_num_boxes = Variable(unsup_num_boxes)
  unsup_gt_boxes = Variable(unsup_gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN_1 = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_2 = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN_1 = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_2 = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN_1 = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_2 = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN_1 = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
    fasterRCNN_2 = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN_1.create_architecture()
  fasterRCNN_2.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params_1 = []
  for key, value in dict(fasterRCNN_1.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params_1 += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params_1 += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
  
  
  params_2 = []
  for key, value in dict(fasterRCNN_2.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params_2 += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params_2 += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]


  #pdb.set_trace()

  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer_1 = torch.optim.Adam(params_1)
    optimizer_2 = torch.optim.Adam(params_2)

  elif args.optimizer == "sgd":
    optimizer_1 = torch.optim.SGD(params_1, momentum=cfg.TRAIN.MOMENTUM)
    optimizer_2 = torch.optim.SGD(params_2, momentum=cfg.TRAIN.MOMENTUM)

  if args.cuda:
    fasterRCNN_1.cuda()
    fasterRCNN_2.cuda()

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_1_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN_1.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
    
    load_name = os.path.join(output_dir,
      'faster_rcnn_2_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN_2.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN_1 = nn.DataParallel(fasterRCNN_1)
    fasterRCNN_2 = nn.DataParallel(fasterRCNN_2)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(args.start_epoch, args.max_epochs + 1):
    # setting to train mode
    fasterRCNN_1.train()
    fasterRCNN_2.train()
    loss_temp = 0
    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    sup_data_iter_1 = iter(sup_dataloader_1)
    sup_data_iter_2 = iter(sup_dataloader_2)
    unsup_data_iter = iter(unsup_dataloader)

    for step in range(iters_per_epoch):

      pdb.set_trace()
      fasterRCNN_1.zero_grad()
      fasterRCNN_2.zero_grad()
      # supervised datum stream 1
      sup_data_1 = next(sup_data_iter_1)
      sup_im_data_1.data.resize_(sup_data_1[0].size()).copy_(sup_data_1[0])
      sup_im_info_1.data.resize_(sup_data_1[1].size()).copy_(sup_data_1[1])
      sup_gt_boxes_1.data.resize_(sup_data_1[2].size()).copy_(sup_data_1[2])
      sup_num_boxes_1.data.resize_(sup_data_1[3].size()).copy_(sup_data_1[3])
      # supervised datum stream 2
      sup_data_2 = next(sup_data_iter_2)
      sup_im_data_2.data.resize_(sup_data_2[0].size()).copy_(sup_data_2[0])
      sup_im_info_2.data.resize_(sup_data_2[1].size()).copy_(sup_data_2[1])
      sup_gt_boxes_2.data.resize_(sup_data_2[2].size()).copy_(sup_data_2[2])
      sup_num_boxes_2.data.resize_(sup_data_2[3].size()).copy_(sup_data_2[3])
      # unsupervised datum stream
      unsup_data = next(unsup_data_iter)
      unsup_im_data.data.resize_(unsup_data[0].size()).copy_(unsup_data[0])
      unsup_im_info.data.resize_(unsup_data[1].size()).copy_(unsup_data[1])
      unsup_gt_boxes.data.resize_(unsup_data[2].size()).copy_(unsup_data[2])
      unsup_num_boxes.data.resize_(unsup_data[3].size()).copy_(unsup_data[3])

      #pdb.set_trace()
      epsilon = 0.1
      #generate adversarial examples for fasterRCNN_1
      ####################################################
      sup_rois_1, sup_cls_prob_1, sup_bbox_pred_1, \
      sup_rpn_loss_cls_1, sup_rpn_loss_box_1, \
      sup_RCNN_loss_cls_1, sup_RCNN_loss_bbox_1, \
      sup_rois_label_1 = fasterRCNN_1(sup_im_data_1, sup_im_info_1, \
                                      sup_gt_boxes_1, sup_num_boxes_1)
      
      sup_loss_1 = sup_rpn_loss_cls_1.mean() + sup_rpn_loss_box_1.mean() \
           + sup_RCNN_loss_cls_1.mean() + sup_RCNN_loss_bbox_1.mean()
      
      sup_loss_1.backward()
      sup_im_data_1_grad = torch.sign(sup_im_data_1.grad.data)
      sup_im_data_1_adv = sup_im_data_1.data + epsilon * sup_im_data_1_grad

      #####################################################
      unsup_rois_1, unsup_cls_prob_1, unsup_bbox_pred_1, \
      unsup_rpn_loss_cls_1, unsup_rpn_loss_box_1, \
      unsup_RCNN_loss_cls_1, unsup_RCNN_loss_bbox_1, \
      unsup_rois_label_1 = fasterRCNN_1(unsup_im_data, unsup_im_info, \
                                unsup_gt_boxes, unsup_num_boxes)

      unsup_loss_1 = unsup_rpn_loss_cls_1.mean() + unsup_rpn_loss_box_1.mean() \
           + unsup_RCNN_loss_cls_1.mean() + unsup_RCNN_loss_bbox_1.mean()

      unsup_loss_1.backward()
      unsup_im_data_1_grad = torch.sign(unsup_im_data.grad.data)
      unsup_im_data_1_adv = unsup_im_data.data + epsilon * unsup_im_data_1_grad
      fasterRCNN_1.zero_grad()
      #####################################################
      
      
      #generate adversarial examples for fasterRCNN_2
      ####################################################
      sup_rois_2, sup_cls_prob_2, sup_bbox_pred_2, \
      sup_rpn_loss_cls_2, sup_rpn_loss_box_2, \
      sup_RCNN_loss_cls_2, sup_RCNN_loss_bbox_2, \
      sup_rois_label_2 = fasterRCNN_2(sup_im_data_2, sup_im_info_2, \
                                      sup_gt_boxes_2, sup_num_boxes_2)
      
      sup_loss_2 = sup_rpn_loss_cls_2.mean() + sup_rpn_loss_box_2.mean() \
           + sup_RCNN_loss_cls_2.mean() + sup_RCNN_loss_bbox_2.mean()
      
      sup_loss_2.backward()
      sup_im_data_2_grad = torch.sign(sup_im_data_2.grad.data)
      sup_im_data_2_adv = sup_im_data_2.data + epsilon * sup_im_data_2_grad

      #####################################################
      unsup_rois_2, unsup_cls_prob_2, unsup_bbox_pred_2, \
      unsup_rpn_loss_cls_2, unsup_rpn_loss_box_2, \
      unsup_RCNN_loss_cls_2, unsup_RCNN_loss_bbox_2, \
      unsup_rois_label_2 = fasterRCNN_1(unsup_im_data, unsup_im_info, \
                                unsup_gt_boxes, unsup_num_boxes)

      unsup_loss_2 = unsup_rpn_loss_cls_2.mean() + unsup_rpn_loss_box_2.mean() \
           + unsup_RCNN_loss_cls_2.mean() + unsup_RCNN_loss_bbox_2.mean()

      unsup_loss_2.backward()
      unsup_im_data_2_grad = torch.sign(unsup_im_data.grad.data)
      unsup_im_data_2_adv = unsup_im_data.data + epsilon * unsup_im_data_2_grad
      fasterRCNN_2.zero_grad()
      #####################################################


      # calculate the supervised losses for fasterRCNN_1
      sup_rois_1, sup_cls_prob_1, sup_bbox_pred_1, \
      sup_rpn_loss_cls_1, sup_rpn_loss_box_1, \
      sup_RCNN_loss_cls_1, sup_RCNN_loss_bbox_1, \
      sup_rois_label_1 = fasterRCNN_1(sup_im_data_1, sup_im_info_1, \
                                sup_gt_boxes_1, sup_num_boxes_1)

      sup_loss_1 = sup_rpn_loss_cls_1.mean() + sup_rpn_loss_box_1.mean() \
           + sup_RCNN_loss_cls_1.mean() + sup_RCNN_loss_bbox_1.mean()

      loss_temp += sup_loss_1.item()
      
      # calculate the supervised losses for fasterRCNN_2
      sup_rois_2, sup_cls_prob_2, sup_bbox_pred_2, \
      sup_rpn_loss_cls_2, sup_rpn_loss_box_2, \
      sup_RCNN_loss_cls_2, sup_RCNN_loss_bbox_2, \
      sup_rois_label_2 = fasterRCNN_2(sup_im_data_2, sup_im_info_2, \
                                sup_gt_boxes_2, sup_num_boxes_2)

      sup_loss_2 = sup_rpn_loss_cls_2.mean() + sup_rpn_loss_box_2.mean() \
           + sup_RCNN_loss_cls_2.mean() + sup_RCNN_loss_bbox_2.mean()

      loss_temp += sup_loss_2.item()


      # calculate the unsupervised jsd loss
      unsup_rois_1, unsup_cls_prob_1, unsup_bbox_pred_1, \
      unsup_rpn_loss_cls_1, unsup_rpn_loss_box_1, \
      unsup_RCNN_loss_cls_1, unsup_RCNN_loss_bbox_1, \
      unsup_rois_label_1 = fasterRCNN_1(unsup_im_data, unsup_im_info, \
                                unsup_gt_boxes, unsup_num_boxes)
      
      unsup_rois_2, unsup_cls_prob_2, unsup_bbox_pred_2, \
      unsup_rpn_loss_cls_2, unsup_rpn_loss_box_2, \
      unsup_RCNN_loss_cls_2, unsup_RCNN_loss_bbox_2, \
      unsup_rois_label_2 = fasterRCNN_1(unsup_im_data, unsup_im_info, \
                                unsup_gt_boxes, unsup_num_boxes)

      # Calculate Jensen-Shannon divergence between unsup_cls_prob_1 & unsup_cls_prob_2
      # JSdiv(p1, p2): 0.5*KLdiv(p1, m) + 0.5*KLdiv(p2, m), where m=0.5*(p1+p2)
      m = 0.5 * (unsup_cls_prob_1 + unsup_cls_prob_2)
      kld_p1_m = (unsup_cls_prob_1 * (unsup_cls_prob_1.log() - m.log())).sum()
      kld_p2_m = (unsup_cls_prob_2 * (unsup_cls_prob_2.log() - m.log())).sum()
      loss_jsd = 0.5 * (kld_p1_m + kld_p2_m)

      loss_temp += loss_jsd.item()


      # Calculate the BCE loss with the real images and adversarial images
      # send the adv images obtained from fasterRCNN_1 to the fasterRCNN_2
      _, sup_cls_prob_advfrom1, _, _, \
      _, _, _, _ = fasterRCNN_2(sup_im_data_1_adv, sup_im_info_1, \
                                sup_gt_boxes_1, sup_num_boxes_1)
      _, unsup_cls_prob_advfrom1, _, _, \
      _, _, _, _ = fasterRCNN_2(unsup_im_data_1_adv, unsup_im_info, \
                                unsup_gt_boxes, unsup_num_boxes)

      # send the adv images obtained from fasterRCNN_2 to the fasterRCNN_1
      _, sup_cls_prob_advfrom2, _, _, \
      _, _, _, _ = fasterRCNN_1(sup_im_data_2_adv, sup_im_info_2, \
                                sup_gt_boxes_2, sup_num_boxes_2)
      _, unsup_cls_prob_advfrom2, _, _, \
      _, _, _, _ = fasterRCNN_1(unsup_im_data_2_adv, unsup_im_info, \
                                unsup_gt_boxes, unsup_num_boxes)

      #  For example,
      #  ask the the prediction of fasterRCNN_1 on im_data_1 to be resistant
      #  to that of fasterRCNN_2 on im_data_1_adv

      pdb.set_trace()
      # 1. sup_cls_prob_1 <--> sup_cls_prob_advfrom1
      loss_dif_sup_12 = sup_cls_prob_advfrom1 * sup_cls_prob_1.log() + (1-sup_cls_prob_advfrom1) * (1-sup_cls_prob_1).log()
      loss_dif_sup_12 = - loss_dif_sup_12.mean()
      # 2. unsup_cls_prob_1 <--> unsup_cls_prob_advfrom1
      loss_dif_unsup_12 = unsup_cls_prob_advfrom1 * unsup_cls_prob_1.log() + (1-unsup_cls_prob_advfrom1) * (1-unsup_cls_prob_1).log()
      loss_dif_unsup_12 = - loss_dif_unsup_12.mean()
      # 3. sup_cls_prob_2 <--> sup_cls_prob_advfrom2
      loss_dif_sup_21 = sup_cls_prob_advfrom2 * sup_cls_prob_2.log() + (1-sup_cls_prob_advfrom2) * (1-sup_cls_prob_2).log()
      loss_dif_sup_21 = - loss_dif_sup_21.mean()
      # 4. unsup_cls_prob_2 <--> unsup_cls_prob_advfrom2
      loss_dif_unsup_21 = unsup_cls_prob_advfrom2 * unsup_cls_prob_2.log() + (1-unsup_cls_prob_advfrom2) * (1-unsup_cls_prob_2).log()
      loss_dif_unsup_21 = - loss_dif_unsup_21.mean()
      
      loss_dif = loss_dif_sup_12 + loss_dif_unsup_12 + loss_dif_sup_21 + loss_dif_unsup_21

      loss_temp += loss_dif.item()

      loss = sup_loss_1 + sup_loss_2 + loss_jsd + loss_dif



      # backward
      pdb.set_trace()
      optimizer_1.zero_grad()
      optimizer_2.zero_grad()
      loss.backward()

      pdb.set_trace()
      if args.net == "vgg16":
          clip_gradient(fasterRCNN_1, 10.)
          clip_gradient(fasterRCNN_2, 10.)
      optimizer_1.step()
      optimizer_2.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = sup_rpn_loss_cls_1.mean().item()
          loss_rpn_box = sup_rpn_loss_box_1.mean().item()
          loss_rcnn_cls = sup_RCNN_loss_cls_1.mean().item()
          loss_rcnn_box = sup_RCNN_loss_bbox_1.mean().item()
          #loss_rpn_cls = rpn_loss_cls.mean().data[0]
          #loss_rpn_box = rpn_loss_box.mean().data[0]
          #loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
          #loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
          fg_cnt = torch.sum(sup_rois_label_1.data.ne(0))
          bg_cnt = sup_rois_label_1.data.numel() - fg_cnt
        else:
          loss_rpn_cls = sup_rpn_loss_cls_1.item()
          loss_rpn_box = sup_rpn_loss_box_1.item()
          loss_rcnn_cls = sup_RCNN_loss_cls_1.item()
          loss_rcnn_box = sup_RCNN_loss_bbox_1.item()
          #loss_rpn_cls = rpn_loss_cls.data[0]
          #loss_rpn_box = rpn_loss_box.data[0]
          #loss_rcnn_cls = RCNN_loss_cls.data[0]
          #loss_rcnn_box = RCNN_loss_bbox.data[0]
          fg_cnt = torch.sum(sup_rois_label_1.data.ne(0))
          bg_cnt = sup_rois_label_1.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
        if args.use_tfboard:
          info = {
            'loss': loss_temp,
            'loss_rpn_cls': loss_rpn_cls,
            'loss_rpn_box': loss_rpn_box,
            'loss_rcnn_cls': loss_rcnn_cls,
            'loss_rcnn_box': loss_rcnn_box
          }
          logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

        loss_temp = 0
        start = time.time()

    
    save_name = os.path.join(output_dir, 'faster_rcnn_1_{}_{}_{}_{}.pth'.format(cfg.TRAIN.PER_SUP, args.session, epoch, step))
    save_checkpoint({
      'session': args.session,
      'epoch': epoch + 1,
      'model': fasterRCNN_1.module.state_dict() if args.mGPUs else fasterRCNN_1.state_dict(),
      'optimizer': optimizer_1.state_dict(),
      'pooling_mode': cfg.POOLING_MODE,
      'class_agnostic': args.class_agnostic,
    }, save_name)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()

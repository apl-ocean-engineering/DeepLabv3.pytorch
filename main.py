import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.autograd import Variable
from torchvision import transforms

import cv2
import time

import deepLabv3.deeplab as deeplab
from deepLabv3.pascal import VOCSegmentation
from deepLabv3.cityscapes import Cityscapes
from deepLabv3.utils import AverageMeter, inter_and_union, create_color_cv_image
from deepLabv3.detector import Detector

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False,
                    help='training mode')
parser.add_argument('--exp', type=str, required=True,
                    help='name of experiment')
parser.add_argument('--gpu', type=int, default=0,
                    help='test time gpu device id')
parser.add_argument('--backbone', type=str, default='resnet101',
                    help='resnet101')
parser.add_argument('--dataset', type=str, default='pascal',
                    help='pascal or cityscapes')
parser.add_argument('--groups', type=int, default=None,
                    help='num of groups for group normalization')
parser.add_argument('--epochs', type=int, default=30,
                    help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=4,
                    help='batch size')
parser.add_argument('--base_lr', type=float, default=0.00025,
                    help='base learning rate')
parser.add_argument('--last_mult', type=float, default=1.0,
                    help='learning rate multiplier for last layers')
parser.add_argument('--scratch', action='store_true', default=False,
                    help='train from scratch')
parser.add_argument('--freeze_bn', action='store_true', default=False,
                    help='freeze batch normalization parameters')
parser.add_argument('--weight_std', action='store_true', default=False,
                    help='weight standardization')
parser.add_argument('--beta', action='store_true', default=False,
                    help='resnet101 beta')
parser.add_argument('--crop_size', type=int, default=513,
                    help='image crop size')
parser.add_argument('--resume', type=str, default=None,
                    help='path to checkpoint to resume from')
parser.add_argument('--workers', type=int, default=4,
                    help='number of data loading workers')
parser.add_argument('--voc_path', type=str, default='data/VOCdevkit',
                    help='Path to VOC dataset')
parser.add_argument('--cityscape_path', type=str, default='data/cityscapes',
                    help='Path to cityscape dataset')
args = parser.parse_args()


def main():
  assert torch.cuda.is_available()

  torch.backends.cudnn.benchmark = True
  model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(
      args.backbone, args.dataset, args.exp)
  if args.dataset == 'pascal':
    dataset = VOCSegmentation(args.voc_path,
        train=args.train, crop_size=args.crop_size)
  elif args.dataset == 'cityscapes':
    dataset = Cityscapes(args.cityscape_path,
        train=args.train, crop_size=args.crop_size)
  else:
    raise ValueError('Unknown dataset: {}'.format(args.dataset))
  if args.backbone == 'resnet101':
    model = getattr(deeplab, 'resnet101')(
        pretrained=(not args.scratch),
        num_classes=len(dataset.CLASSES),
        num_groups=args.groups,
        weight_std=args.weight_std,
        beta=args.beta)
  else:
    raise ValueError('Unknown backbone: {}'.format(args.backbone))


  detector = Detector(model)
  if args.train:
    detector.train(dataset, model_fname, args)

  else:
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    model.eval()
    checkpoint = torch.load(model_fname % args.epochs)
    state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k}
    #print(state_dict.keys())
    model.load_state_dict(state_dict)
    cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
    cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

    inter_meter = AverageMeter()
    union_meter = AverageMeter()


    for i in range(len(dataset)):
      prev_time = time.time()
      inputs, target, fname = dataset[i]

      pred = detector.inference(inputs)

      print("time elapsed", time.time() - prev_time)

    iou = inter_meter.sum / (union_meter.sum + 1e-10)
    for i, val in enumerate(iou):
      print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
    print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
  main()

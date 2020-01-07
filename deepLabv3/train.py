import os
import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from deepLabv3.utils import AverageMeter


class Detector:
    def __init__(self, model):
        self.model = model

    def train(self, dataset, model_fname, args):
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.model = nn.DataParallel(self.model).cuda()
        self.model.train()
        if args.freeze_bn:
          for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
              m.eval()
              m.weight.requires_grad = False
              m.bias.requires_grad = False
        backbone_params = (
            list(self.model.module.conv1.parameters()) +
            list(self.model.module.bn1.parameters()) +
            list(self.model.module.layer1.parameters()) +
            list(self.model.module.layer2.parameters()) +
            list(self.model.module.layer3.parameters()) +
            list(self.model.module.layer4.parameters()))
        last_params = list(self.model.module.aspp.parameters())
        optimizer = optim.SGD([
          {'params': filter(lambda p: p.requires_grad, backbone_params)},
          {'params': filter(lambda p: p.requires_grad, last_params)}],
          lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
        dataset_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=args.train,
            pin_memory=True, num_workers=args.workers)
        max_iter = args.epochs * len(dataset_loader)
        losses = AverageMeter()
        start_epoch = 0

        print(dataset_loader)

        if args.resume:
          if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(
              args.resume, checkpoint['epoch']))
          else:
            print('=> no checkpoint found at {0}'.format(args.resume))

        for epoch in range(start_epoch, args.epochs):
          for i, (inputs, target, x) in enumerate(dataset_loader):
            cur_iter = epoch * len(dataset_loader) + i
            lr = args.base_lr * (1 - float(cur_iter) / max_iter) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * args.last_mult

            inputs = Variable(inputs.cuda())
            target = Variable(target.cuda())
            outputs = self.model(inputs)
            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
              pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('epoch: {0}\t'
                  'iter: {1}/{2}\t'
                  'lr: {3:.6f}\t'
                  'loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                  epoch + 1, i + 1, len(dataset_loader), lr, loss=losses))

          if epoch % 10 == 9:
            torch.save({
              'epoch': epoch + 1,
              'state_dict': self.model.state_dict(),
              'optimizer': optimizer.state_dict(),
              }, model_fname % (epoch + 1))

    def inference(self, inputs):
      inputs = Variable(inputs.cuda())
      outputs = self.model(inputs.unsqueeze(0))
      y, pred = torch.max(outputs, 1)
      pred = pred.data.cpu().numpy().squeeze().astype(np.uint8) #indies of classes

      return pred

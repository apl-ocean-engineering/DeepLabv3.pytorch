import numpy as np
import torch
from scipy.io import loadmat

import cv2
import time

import deepLabv3.deeplab as deeplab
from deepLabv3.pascal import VOCSegmentation
from deepLabv3.cityscapes import Cityscapes
from deepLabv3.utils import AverageMeter, inter_and_union, load_model
from deepLabv3.detector import Detector
from deepLabv3.argLoader import ArgLoader


def main():
    assert torch.cuda.is_available()
    argloader = ArgLoader()
    args = argloader.args

    torch.backends.cudnn.benchmark = True
    if args.dataset == 'pascal':
        dataset = VOCSegmentation(
                    args.voc_path,
                    train=args.train, crop_size=args.crop_size)
    elif args.dataset == 'cityscapes':
        dataset = Cityscapes(
                args.cityscape_path,
                train=args.train, crop_size=args.crop_size)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    model, model_fname = load_model(args, dataset.CLASSES)
    detector = Detector(model)
    if args.train:
        detector.train(dataset, model_fname, args)

    else:
        torch.cuda.set_device(args.gpu)
        model = model.cuda()
        model.eval()
        checkpoint = torch.load(model_fname % args.epochs)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()
                      if 'tracked' not in k}

        model.load_state_dict(state_dict)
        cmap = loadmat('data/pascal_seg_colormap.mat')['colormap']
        cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

        inter_meter = AverageMeter()
        union_meter = AverageMeter()

        for i in range(len(dataset)):
            prev_time = time.time()

            inputs, target, fname = dataset[i]

            pred = detector.inference(inputs)

            mask = target.numpy().astype(np.uint8)

            inter, union = inter_and_union(pred, mask, len(dataset.CLASSES))
            inter_meter.update(inter)
            union_meter.update(union)

            print("time elapsed", time.time() - prev_time)

        iou = inter_meter.sum / (union_meter.sum + 1e-10)
        for i, val in enumerate(iou):
            print('IoU {0}: {1:.2f}'.format(dataset.CLASSES[i], val * 100))
        print('Mean IoU: {0:.2f}'.format(iou.mean() * 100))


if __name__ == "__main__":
    main()

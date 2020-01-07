import argparse


class ArgLoader:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='deeplabv3 \
                                              Detection Module')
        self.parser.add_argument(
                            '--train', action='store_true', default=False,
                            help='training mode')
        self.parser.add_argument(
                            '--exp', type=str, required=True,
                            help='name of experiment')
        self.parser.add_argument(
                            '--gpu', type=int, default=0,
                            help='test time gpu device id')
        self.parser.add_argument(
                            '--backbone', type=str, default='resnet101',
                            help='resnet101')
        self.parser.add_argument(
                            '--dataset', type=str, default='pascal',
                            help='pascal or cityscapes')
        self.parser.add_argument(
                            '--groups', type=int, default=None,
                            help='num of groups for group normalization')
        self.parser.add_argument(
                            '--epochs', type=int, default=30,
                            help='num of training epochs')
        self.parser.add_argument(
                            '--batch_size', type=int, default=4,
                            help='batch size')
        self.parser.add_argument(
                            '--base_lr', type=float, default=0.00025,
                            help='base learning rate')
        self.parser.add_argument(
                            '--last_mult', type=float, default=1.0,
                            help='learning rate multiplier for last layers')
        self.parser.add_argument(
                            '--scratch', action='store_true', default=False,
                            help='train from scratch')
        self.parser.add_argument(
                            '--freeze_bn', action='store_true', default=False,
                            help='freeze batch normalization parameters')
        self.parser.add_argument(
                            '--weight_std', action='store_true', default=False,
                            help='weight standardization')
        self.parser.add_argument(
                            '--beta', action='store_true', default=False,
                            help='resnet101 beta')
        self.parser.add_argument(
                            '--crop_size', type=int, default=513,
                            help='image crop size')
        self.parser.add_argument(
                            '--resume', type=str, default=None,
                            help='path to checkpoint to resume from')
        self.parser.add_argument(
                            '--workers', type=int, default=4,
                            help='number of data loading workers')
        self.parser.add_argument(
                            '--voc_path', type=str, default='data/VOCdevkit',
                            help='Path to VOC dataset')
        self.parser.add_argument(
                            '--cityscape_path', type=str,
                            default='data/cityscapes',
                            help='Path to cityscape dataset')

        self.parser.add_argument(
                            '--display', type=bool, default=False,
                            help='display detected image')

    @property
    def args(self):
        return self.parser.parse_args()

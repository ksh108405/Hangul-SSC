"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import HANGUL_ROOT, HangulDetection, HANGUL_CLASSES
from data import BaseTransform
import torch.utils.data as data
from sys import stdout

from ssc import build_ssc

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import shutil


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/v4/SSC150_Hangul_v4_full.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset', default='Hangul',
                    help='Name of dataset to evaluate')
parser.add_argument('--network_size', default=150, type=int,
                    help='SSC network size (only 150, 300, 512 and 1024 are supported)')
parser.add_argument('--print_false', default=True, type=str2bool,
                    help='Print list of image name which model not classified correctly.')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def test_net(net, dataset):
    num_images = len(dataset)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    true = []
    false = []

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        prediction = torch.argmax(detections)

        if int(prediction) == gt[0]:
            true += [i]
        else:
            false += [i]

        if i % 1 == 0:
            stdout.write('\rim_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
            stdout.flush()
    print()
    print(f'Accuracy : {len(true)/(len(true)+len(false))}')
    print('list of false images : ')
    for image_name in sorted(list(map(dataset.get_image_name, false))):
        print(image_name)




if __name__ == '__main__':
    # load net
    if args.dataset == 'Hangul':
        imgpath = os.path.join(HANGUL_ROOT, 'test', '%s.png')
        dataset_path = HANGUL_ROOT
        dataset_mean = (104, 117, 123)
        set_type = 'test'
        labelmap = HANGUL_CLASSES
        dataset_root = HANGUL_ROOT
    else:
        raise Exception('Please specify correct dataset name!')

    num_classes = len(labelmap)
    net = build_ssc('test', args.network_size, num_classes, cfg=args.dataset)  # initialize SSC
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    if args.dataset == 'Hangul':
        dataset = HangulDetection(dataset_root, 'test',
                                  BaseTransform(args.network_size, dataset_mean))
    else:
        raise Exception('Please specify correct dataset name!')

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(net, dataset)

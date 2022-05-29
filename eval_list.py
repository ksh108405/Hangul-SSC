from eval import test_net
from ssc import build_ssc
from data import HANGUL_ROOT, HangulDetection, HANGUL_CLASSES
from data import BaseTransform
import os
import argparse
import torch
import torch.backends.cudnn as cudnn


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model_dir',
                    default='./weights/v2/', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--dataset', default='Hangul',
                    help='Name of dataset to evaluate')
parser.add_argument('--network_size', default=150, type=int,
                    help='SSC network size (only 150, 300, 512 and 1024 are supported)')

args = parser.parse_args()

if args.dataset == 'Hangul':
    labelmap = HANGUL_CLASSES
    dataset_root = HANGUL_ROOT
else:
    raise Exception('Please specify correct dataset name.')

num_classes = len(labelmap)
model_list = os.listdir(args.trained_model_dir)
model_list.sort()
for weight_name in model_list:
    net = build_ssc('test', args.network_size, num_classes, cfg='Hangul')  # initialize SSC
    net.load_state_dict(torch.load(args.trained_model_dir + weight_name))
    net.eval()
    print(f'Model = {weight_name}')
    dataset = HangulDetection(dataset_root, 'test',
                              BaseTransform(150, (0, 0, 0)))

    net = net.cuda()
    cudnn.benchmark = True
    # evaluation
    test_net(net, dataset)
    del net

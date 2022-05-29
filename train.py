from data import *
from utils.augmentations import SSCAugmentation
from ssc import build_ssc
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import torch.onnx
import argparse
from torchinfo import summary

"""
from torch.multiprocessing import set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
"""


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiFeature Classifier Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='Hangul',
                    type=str, help='Which dataset to train')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=1, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=6, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for Adam')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for learning rate')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--weight_name', default='SSC150_Hangul_v5',
                    help='Saved weight name')
parser.add_argument('--train_set', default='train',
                    help='used for divide train or test')
parser.add_argument('--augmentation', default=True,
                    help='Whether to take augmentation process.')
parser.add_argument('--shuffle', default=True,
                    help='set to True to have the data reshuffled at every epoch.')
parser.add_argument('--one_epoch', default=False,
                    help='Only iterate for one epoch.')
parser.add_argument('--save_onnx_model', default=False,
                    help='Save ONNX model file.')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'Hangul':
        cfg = hangul
        image_sets = 'train'
        dataset = HangulDetection(root=HANGUL_ROOT, image_sets=image_sets,
                                  transform=SSCAugmentation(cfg['min_dim'], MEANS, args.augmentation))
    else:
        raise Exception('Only hangul dataset is available.')

    ssc_net = build_ssc('train', cfg['min_dim'], cfg['num_classes'], args.dataset)

    net = ssc_net

    if args.cuda:
        net = torch.nn.DataParallel(ssc_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssc_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        ssc_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssc_net.extras.apply(weights_init)
        ssc_net.linear.apply(weights_init)
        ssc_net.final.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    net.train()

    if args.save_onnx_model:
        summary(ssc_net, input_size=(args.batch_size, 3, cfg['min_dim'], cfg['min_dim']), device='cuda')
        dummy_data = torch.empty(args.batch_size, 3, 150, 150, dtype=torch.float32)
        torch.onnx.export(ssc_net, dummy_data, "output.onnx", opset_version=7)
        print('ONNX saved. exiting...')
        exit(0)

    # loss counters
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSC on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    # adjust lr on resuming training
    if args.start_iter != 1:
        lr_adjusted = False
    else:
        lr_adjusted = True

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=args.shuffle, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter'] + 1):

        # adjust lr on resuming training
        if not lr_adjusted:
            for i, lr_step in enumerate(reversed(cfg['lr_steps'])):
                if iteration > lr_step:
                    step_index = len(cfg['lr_steps']) - i
                    adjust_learning_rate(optimizer, args.gamma, step_index)
                    lr_adjusted = True
                    break

        if iteration - 1 in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = torch.tensor([Variable(ann) for ann in targets]).cuda().to(dtype=torch.int64)
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = torch.tensor([Variable(ann) for ann in targets]).to(dtype=torch.int64)

        # debug = images.cpu().numpy()

        # forward
        t0 = time.time()
        out = net(images)
        # backprop
        optimizer.zero_grad()

        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if iteration % 100 == 0:
            print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.data + ' timer: %.4f sec.' % (t1 - t0))
            """
            for i, para in enumerate(net.parameters()):
                print(f'{i + 1}th parameter tensor:', para.shape)
                # print(para)
                print('---')
                print(para.grad)
            """

        if iteration != 0 and iteration % 10000 == 0:
            print('Saving state, iter:', iteration)
            weight_path = args.save_folder + args.weight_name + '_' + repr(iteration)
            while os.path.isfile(weight_path + '.pth'):
                weight_path += '_dup'
            torch.save(ssc_net.state_dict(), weight_path + '.pth')

        if args.one_epoch and iteration > epoch_size:
            print('One epoch reached: exiting training...')
            return 0
    torch.save(ssc_net.state_dict(),
               args.save_folder + args.weight_name + '_full.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** step)
    print(f'adjusting learning rate to {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()

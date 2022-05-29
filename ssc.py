import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import hangul
import os


class SSC(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, final, num_classes, cfg):
        super(SSC, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.size = size

        # SSC network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.linear = nn.ModuleList(head)
        self.final = nn.ModuleList(final)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply fully-connected layer to source layers
        for (x, l) in zip(sources, self.linear):
            x = torch.flatten(x, 1, 3)
            x = F.relu(l(x), inplace=True)
            conf.append(x)

        conf = torch.cat([o for o in conf], 1)

        # apply final fully-connected layer
        for v in self.final:
            conf = v(conf)

        # print(conf)

        if self.phase == "test":
            output = self.softmax(conf)
        else:
            output = conf
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S' and in_channels != 'F':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            elif v == 'F':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=4, stride=1, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


def multibox(vgg, extra_layers, num_classes, cfg):
    linear_layers = []
    vgg_source = [-2]
    cfg_idx = 0
    for k, v in enumerate(vgg_source):
        linear_layers += [nn.Linear(cfg[cfg_idx] * cfg[cfg_idx] * vgg[v].out_channels, num_classes)]
        cfg_idx += 1
    for k, v in enumerate(extra_layers[1::2], 2):
        linear_layers += [nn.Linear(cfg[cfg_idx] * cfg[cfg_idx] * v.out_channels, num_classes)]
        cfg_idx += 1
    final_linear_layer = [nn.Linear(len(linear_layers) * num_classes, len(linear_layers) * num_classes), nn.ReLU(inplace=True), nn.Linear(len(linear_layers) * num_classes, num_classes), nn.ReLU(inplace=True), nn.Linear(num_classes, num_classes)]
    return vgg, extra_layers, linear_layers, final_linear_layer


base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
extras = {
    '150': [256, 'S', 512, 128, 'S', 256, 128, 256],
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'F', 256],
    '1024': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'F', 256]
}


def build_ssc(phase, size=150, num_classes=21, cfg=None):
    if phase != "test" and phase != "train":
        raise Exception("ERROR: Phase: " + phase + " not recognized")
    if size not in [150, 300, 512, 1024]:
        raise Exception("ERROR: You specified size " + repr(size) + ". However, " +
                        "currently only SSC300, SSC512 and SSC1024 are supported!")

    if cfg == 'Hangul':
        cfg = hangul
    else:
        raise Exception("ERROR: You specified config " + str(cfg) + ". However, " +
                        "currently hangul data are only supported!")

    base_, extras_, head_, final_ = multibox(vgg(base, 3),
                                     add_extras(extras[str(size)], 1024),
                                     num_classes, cfg['feature_maps'])

    return SSC(phase, size, base_, extras_, head_, final_, num_classes, cfg)

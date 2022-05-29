import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import imgaug as ia
import imgaug.augmenters as iaa


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels

    def print(self):
        return self.transforms


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, labels=None):
        return self.lambd(img, labels)


class ConvertFromInts(object):
    def __call__(self, image, labels=None):
        return image.astype(np.float32), labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, labels=None):
        image = cv2.resize(image, (self.size,
                                 self.size))
        return image, labels


class ToCV2Image(object):
    def __call__(self, tensor, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), labels


class ToTensor(object):
    def __call__(self, cvimage, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), labels


class SSCAugmentation(object):
    def __init__(self, size=150, mean=(104, 117, 123), augment=True):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            Resize(self.size),
            # SubtractMeans(self.mean)
        ])
        if augment is True:
            print('Augmentation turned on.')
            self.seq = iaa.Sequential(
                [
                    sometimes(iaa.Affine(
                        scale={"x": (0.8, 1.0), "y": (0.8, 1.0)},
                        # scale images to 80-100% of their size, individually per axis
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        # translate by -10 to +10 percent (per axis)
                        rotate=(-10, 10),  # rotate by -10 to +10 degrees
                        shear=(-10, 10),  # shear by -10 to +10 degrees
                        cval=0,  # if mode is constant, use a constant value 0
                        mode='constant',
                        fit_output=True
                    )),
                    # execute 0 to 3 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((0, 3),
                               [
                                   iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
                                   # add gaussian noise to images
                                   iaa.Add((-10, 10)),
                                   # change brightness of images (by -10 to 10 of original value)
                                   iaa.LinearContrast((0.5, 1.25)),
                                   # improve or worsen the contrast
                                   iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                                   # move parts of the image around
                                   iaa.ElasticTransformation(alpha=(1.0, 15.5), sigma=4.0),
                                   # move pixels locally around (with random strengths)
                                   iaa.PerspectiveTransform(scale=(0.01, 0.05), fit_output=True)
                               ],
                               random_order=True
                               )
                ],
                random_order=True
            )
        else:
            print('Augmentation turned off.')
            self.seq = None
        print(self.seq)

    def __call__(self, img, labels):
        if self.seq is not None:
            img = self.seq(image=img)
        img, labels = self.augment(img, labels)
        return img, labels

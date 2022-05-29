"""Hangul Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot, Seok-Hoon Kang
"""
from .config import HOME
import os
import os.path as osp
import torch
import torch.utils.data as data
import cv2

HANGUL_CLASSES = (  # always index 0
    'AC00', 'AC1C', 'AC70', 'AC8C', 'ACA8', 'ACC4', 'AD50', 'AD6C', 'ADDC', 'ADF8', 'AE30', 'AE4C', 'AE68', 'AED8',
    'AFB8', 'B07C', 'B098', 'B0B4', 'B108', 'B124', 'B140', 'B204', 'B290', 'B2C8', 'B2E4', 'B300', 'B354', 'B370',
    'B450', 'B4DC', 'B514', 'B530', 'B54C', 'B5A0', 'B77C', 'B798', 'B7EC', 'B808', 'B824', 'B840', 'B8CC', 'B8E8',
    'B958', 'B974', 'B9AC', 'B9C8', 'B9E4', 'BA38', 'BA54', 'BA70', 'BB34', 'BBC0', 'BBF8', 'BC14', 'BC30', 'BC84',
    'BCA0', 'BD80', 'BE0C', 'BE44', 'BE60', 'C0AC', 'C0C8', 'C11C', 'C138', 'C218', 'C2A4', 'C2DC', 'C2F8', 'C368',
    'C4F0', 'C528', 'C544', 'C560', 'C57C', 'C5B4', 'C5D0', 'C5EC', 'C608', 'C694', 'C6B0', 'C720', 'C73C', 'C774',
    'C790', 'C7AC', 'C800', 'C81C', 'C838', 'C8FC', 'C988', 'C9C0', 'C9F8', 'CC28', 'CC44', 'CC98', 'CCB4', 'CCD0',
    'CD94', 'CE58', 'CE74', 'CEE4', 'CF00', 'CF1C', 'D06C', 'D0A4', 'D0C0', 'D0DC', 'D130', 'D14C', 'D22C', 'D2B8',
    'D2F0', 'D30C', 'D328', 'D37C', 'D3D0', 'D45C', 'D478', 'D504', 'D53C', 'D558', 'D574', 'D5C8', 'D600', 'D6A8',
    'D6C4', 'D788'
)

# note: if you used our download scripts, this should be right
HANGUL_ROOT = osp.join(HOME, "data/hil-seri/")


class HangulDetection(data.Dataset):
    """Hangul Detection Dataset Object

    input is image, target is lable index.

    Arguments:
        root (string): filepath to HIL-SERI folder.
        image_set (string): imageset to use (eg. 'train', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'HIL-SERI')
    """

    def __init__(self, root,
                 image_sets='train',
                 transform=None,
                 dataset_name='HIL-SERI'):
        self.root = root
        self.image_set = image_sets
        self._imgpath = osp.join(root, image_sets)
        self.transform = transform
        self.name = dataset_name
        self.ids = list()
        self.class_to_ind = dict(zip(HANGUL_CLASSES, range(len(HANGUL_CLASSES))))
        for image_name in os.listdir(self._imgpath):
            self.ids.append(image_name)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        img = cv2.imread(osp.join(self._imgpath, img_id))
        height, width, channels = img.shape

        class_idx = self.class_to_ind[img_id.split('_')[2][:-4]]

        if self.transform is not None:
            img, labels = self.transform(img, class_idx)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
        return torch.from_numpy(img).permute(2, 0, 1), [class_idx], height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def get_image_name(self, index):
        return self.ids[index]

import os
import shutil
import cv2
from data.hangul import HANGUL_ROOT

datasets_path = [os.path.join(HANGUL_ROOT, 'train'), os.path.join(HANGUL_ROOT, 'test')]
backup_path = os.path.join(HANGUL_ROOT, 'backup')

excluded_train_path = "excluded train set.txt"
with open(excluded_train_path) as f:
    excluded_train = f.read().splitlines()

excluded_train_path = "excluded test set.txt"
with open(excluded_train_path) as f:
    excluded_test = f.read().splitlines()

excluded_lists = [excluded_train, excluded_test]

for i, dataset_path in enumerate(datasets_path):
    for train_image in os.listdir(dataset_path):
        if train_image.split('.')[0] not in excluded_lists[i]:
            continue
        shutil.move(os.path.join(dataset_path, train_image), os.path.join(backup_path, train_image))

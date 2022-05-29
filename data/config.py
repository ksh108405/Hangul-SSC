# config.py
import os.path

size_index = {'150': 0, '300': 1, '512': 2, '1024': 3}

# gets home dir cross platform
HOME = os.path.expanduser("~")

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# SSC150, SSC300, SSC512, SSC1024 CONFIGS
HANGUL_NETWORK_SIZE = 150  # 150 or 300 or 512 or 1024

hangul = {
    'num_classes': 128,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [[9, 5, 3, 1], [19, 10, 5, 3, 1], [32, 16, 8, 4, 2, 1], [64, 32, 16, 8, 4, 2, 1]][
        size_index[str(HANGUL_NETWORK_SIZE)]],
    'min_dim': 150,
    'clip': True,
    'name': 'HIL-SERI'
}

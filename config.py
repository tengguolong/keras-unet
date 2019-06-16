# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:45:59 2019

@author: Teng
"""

import numpy as np
from easydict import EasyDict as edict

C = edict()
cfg = C

C.classes = ('background', 'aeroplane', 'bicycle', 'bird', 'boat','bottle',
             'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
             'motorbike', 'person', 'pottedplant','sheep', 'sofa', 'train', 'tvmonitor')

C.n_classes = len(C.classes)

C.mean = np.array([[[123.68, 116.779, 103.939]]], dtype=np.float32)

C.height = 572
C.width = 572
C.seg_height = 388
C.seg_width = 388

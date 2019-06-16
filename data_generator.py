# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:56:00 2019

@author: Teng
"""

from __future__ import print_function
from keras.utils import Sequence
import numpy as np
import cv2
from PIL import Image
import os.path as osp
from config import cfg
    


def augmentation(image, label):
    def rnd_bright(img):
        delta = np.random.uniform(-32, 32)
        return img+delta
    
    def channel_swap(img):
        channels = [(0,1,2), (0,2,1), (1,0,2), (1,2,0), (2,0,1), (2,1,0)]
        order = channels[np.random.randint(6)]
        return img[:, :, order]
    # 水平翻转
    if np.random.randint(2):
        image = image[:, ::-1, :]
        label = label[:, ::-1]
    # 随机亮度调整
#    image = rnd_bright(image)
    # 随机通道交换
#    image = channel_swap(image)
    
    return [image, label]


class DataGenerator(Sequence):
    def __init__(self,
                 split='train',
                 batch_size=1,
                 aug=True,
                 shuffle=True):
        self.split = split
        self.batch_size = batch_size
        self.aug = aug
        self.shuffle = shuffle
        self.root_dir = osp.abspath('../data/VOCdevkit/VOC2012')
        self.img_dir = osp.join(self.root_dir, 'JPEGImages')
        self.seg_dir = osp.join(self.root_dir, 'SegmentationClass')
        self.names = self.load_image_names()
        self.indexes = np.arange(len(self.names))
        self.on_epoch_end()

    def __len__(self):
        '''Denote the number of batches per epoch'''
        return int(np.floor(len(self.names)/self.batch_size))

    def __getitem__(self, index):
        '''generate one batch of data'''
        inds = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        blob = self.data_generation(inds)
        return blob

    def on_epoch_end(self):
        '''updata index after each epoch'''
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def load_image_names(self):
        path = osp.join(self.root_dir, 'ImageSets', 'Segmentation', self.split+'.txt')
        with open(path) as f:
            names = [x.strip() for x in f.readlines()]
        return names
    
    def load_image(self, idx):
        """
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        """
        img = Image.open(osp.join(self.img_dir, self.names[idx]+'.jpg'))
        img = np.array(img, dtype=np.float32)
        img = img[:,:,::-1]
        img -= cfg.mean
        img = cv2.resize(img, (cfg.width, cfg.height))
#        img = np.expand_dims(img, 0)
        return img


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = Image.open(osp.join(self.seg_dir, self.names[idx]+'.png'))
        label = np.array(label, dtype=np.uint8)
        label = cv2.resize(label, (cfg.width, cfg.height), interpolation=cv2.INTER_NEAREST)
#        label = np.expand_dims(label, 0)
        return label
    
    def data_generation(self, inds):
        inputs, targets = [], []
        for idx in inds:
            image = self.load_image(idx)
            label = self.load_label(idx)
            if self.aug:
                image, label = augmentation(image, label)
            inputs.append(image)
            targets.append(label)
        
        inputs = np.array(inputs)
        targets = np.array(targets)
        return inputs, targets


if __name__ == '__main__':
    import vis
    palette = vis.make_palette(21)
    g = DataGenerator(split='train', batch_size=1, aug=True, shuffle=False)
    for i in range(0,200,10):
        image, label = g[i]
        image = np.array(image[0] + cfg.mean, 'uint8')
        label = label[0]
        label[np.where(label==255)] = 0
        out_im = Image.fromarray(vis.color_seg(label, palette))
        out_im.save('check_data/{}.png'.format(i))
        masked_im = vis.vis_seg(image, label, palette)
        cv2.imwrite('check_data/{}.jpg'.format(i), masked_im)
    
    
    
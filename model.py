# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:19:04 2019

@author: Teng
"""

from __future__ import print_function
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D, Activation
from keras.layers import Conv2DTranspose, Cropping2D, BatchNormalization
from keras.models import Model
import keras.backend as K

from config import cfg


def crop(x, y):
    h1, w1 = K.int_shape(x)[1:3]
    h2, w2 = K.int_shape(y)[1:3]
    ch = abs(h1 - h2)
    cw = abs(w1 - w2)
    
    if h1 > h2:
        x = Cropping2D(cropping=((ch//2, ch//2),(0,0)), data_format="channels_last")(x)
    elif h1 < h2:
        y = Cropping2D(cropping=((ch//2, ch//2),(0,0)), data_format="channels_last")(y)
    
    if w1 > w2:
        x = Cropping2D(cropping=((0,0),(cw//2, cw//2)), data_format="channels_last")(x)
    elif w1 < w2:
        y = Cropping2D(cropping=((0,0),(cw//2, cw//2)), data_format="channels_last")(y)
    
    return [x, y]            


def contracting_block(input_tensor, n_channels, block_id):
    bn_axis = -1
    x = Conv2D(n_channels, (3, 3), activation=None, padding='same', name='block{}_conv1'.format(block_id))(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(n_channels, (3, 3), activation=None, padding='same', name='block{}_conv2'.format(block_id))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    f = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block{}_pool'.format(block_id))(x)
    return [x, f]


def expansive_block(input_tensor, f, n_channels, block_id):
    bn_axis = -1
    x = Conv2DTranspose(n_channels, kernel_size=(2,2), strides=(2,2), use_bias=False, name='upconv'+block_id)(input_tensor)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x, f = crop(x, f)
    x = Concatenate(name='concat'+block_id)([f, x])
    x = Conv2D(n_channels, (3, 3), padding='same', name='expa{}_conv1'.format(block_id))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(n_channels, (3, 3), padding='same', name='expa{}_conv2'.format(block_id))(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    return x
    

def unet(img_input, n_classes):
    c1, c2, c3, c4, c5 = cfg.channels
    bn_axis = -1
    x, f1 = contracting_block(img_input, c1, '1')
    x, f2 = contracting_block(x, c2, '2')
    x, f3 = contracting_block(x, c3, '3')
    x, f4 = contracting_block(x, c4, '4')

    x = Conv2D(c5, (3, 3), activation=None, padding='same', dilation_rate=(5,5), name='block5_conv1')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    x = Conv2D(c5, (3, 3), activation=None, padding='same', name='block5_conv2')(x)
    x = BatchNormalization(axis=bn_axis)(x)
    x = Activation('relu')(x)
    
    x = expansive_block(x, f4, c4, '1')   
    x = expansive_block(x, f3, c3, '2')   
    x = expansive_block(x, f2, c2, '3')    
    x = expansive_block(x, f1, c1, '4')   
    # output
    x = Conv2D(n_classes, (1, 1), activation='relu', padding='same', name='seg')(x)
    
    return x

    
def unet_model(height, width, n_classes=21):
    i = Input(shape=(height, width, 3))
    o = unet(i, n_classes)
    model = Model(i, o)
    model.name = 'U-Net'
    return model    
    
    
if __name__ == '__main__':
    model = unet_model(320, 320)
    print(model.summary())

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:19:04 2019

@author: Teng
"""

from __future__ import print_function
from keras.layers import Input, Conv2D, Concatenate, MaxPooling2D
from keras.layers import Conv2DTranspose, Cropping2D
from keras.models import Model
import keras.backend as K


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


def unet(img_input, n_classes):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='block1_conv2')(x)
    f1 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x)
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='block2_conv2')(x)
    f2 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block2_pool')(x)
    
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='block3_conv2')(x)
    f3 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block3_pool')(x)
    
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='block4_conv2')(x)
    f4 = x
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='valid', name='block4_pool')(x)
    
    # Block 5
    x = Conv2D(1024, (3, 3), activation='relu', padding='valid', name='block5_conv1')(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='valid', name='block5_conv2')(x)
    
    # add f4
    x = Conv2DTranspose(512, kernel_size=(2,2), strides=(2,2), use_bias=False, name='upconv1')(x)
    x, f4 = crop(x, f4)
    x = Concatenate(name='concat1')([f4, x])
    x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='expa1_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='valid', name='expa1_conv2')(x)
    
    # add f3
    x = Conv2DTranspose(256, kernel_size=(2,2), strides=(2,2), use_bias=False, name='upconv2')(x)
    x, f3 = crop(x, f3)
    x = Concatenate(name='concat2')([f3, x])
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='expa2_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='valid', name='expa2_conv2')(x)
    
    # add f2
    x = Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2), use_bias=False, name='upconv3')(x)
    x, f2 = crop(x, f2)
    x = Concatenate(name='concat3')([f2, x])
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='expa3_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='valid', name='expa3_conv2')(x)
    
    # add f1
    x = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2), use_bias=False, name='upconv4')(x)
    x, f1 = crop(x, f1)
    x = Concatenate(name='concat4')([f1, x])
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='expa4_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='valid', name='expa4_conv2')(x)
    
    # output
    x = Conv2D(n_classes, (1, 1), activation='relu', padding='valid', name='seg')(x)
    
    return x

    
def unet_model(height, width, n_classes=21):
    i = Input(shape=(height, width, 3))
    o = unet(i, n_classes)
    model = Model(i, o)
    model.name = 'U-Net'
    return model    
    
    
if __name__ == '__main__':
    model = unet_model(572, 572)
    print(model.summary())

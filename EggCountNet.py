# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: Weidi Xie

Adapted by Kevin Yost

@description:
This is the file to create the model, similar as the paper, but with batch normalization, make it more easier to train.

TODO: Fix Conv2D errors - import from layers and re-write model.
"""

import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential, Model
from keras.layers import Input, Activation, UpSampling2D
from keras.layers.merge import Concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from data import load_data


class EggCountNet(object):
    def __init__(self, img_rows=480, img_cols=640):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def buildModel_U_net(self):
        input_ = Input((self.img_rows, self.img_cols,3))
        # =========================================================================
        def _conv_bn_relu_x2(nb_filter, row, col, subsample=(1, 1)):
            weight_decay = 1e-5
            def f(input):
                conv_a = Convolution2D(nb_filter, row, col, subsample=subsample,
                                       init='orthogonal', border_mode='same', bias=False,
                                       W_regularizer=l2(weight_decay),
                                       b_regularizer=l2(weight_decay))(input)
                norm_a = BatchNormalization()(conv_a)
                act_a = Activation(activation='relu')(norm_a)
                conv_b = Convolution2D(nb_filter, row, col, subsample=subsample,
                                       init='orthogonal', border_mode='same', bias=False,
                                       W_regularizer=l2(weight_decay),
                                       b_regularizer=l2(weight_decay))(act_a)
                norm_b = BatchNormalization()(conv_b)
                act_b = Activation(activation='relu')(norm_b)
                return act_b
            return f

        nb_filter = 64
        block1 = _conv_bn_relu_x2(nb_filter, 3, 3)(input)  # filters: -> 64
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        nb_filter *= 2  # default nb_filter = 128
        block2 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool1)  # filters: 64 -> 128
        pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
        # =========================================================================
        nb_filter *= 2  # default nb_filter = 256
        block3 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool2)  # filters: 128 -> 256
        pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
        # =========================================================================
        nb_filter *= 2  # default nb_filter = 512
        block4 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool3)  # filters: 256 -> 512
        pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
        # =========================================================================
        nb_filter *= 2  # default nb_filter = 1024
        block5 = _conv_bn_relu_x2(nb_filter, 3, 3)(pool4)   # filters: 512 -> 1024
        up_conv5 = UpSampling2D(size=(2, 2))(block5)        # filters: 1024 -> 512
        up5 = Concatenate([block4, up_conv5], axis=-1)      # concat: 512 + 512 = 1024
        # =========================================================================
        nb_filter /= 2  # default nb_filter = 512
        block6 = _conv_bn_relu_x2(nb_filter, 3, 3)(up5)     # filters: 1024 -> 512
        up_conv6 = UpSampling2D(size=(2, 2))(block6)        # filters: 512 -> 256
        up6 = Concatenate([block3, up_conv6], axis=-1)      # concat: 256 + 256 = 512
        # =========================================================================
        nb_filter /= 2  # default nb_filter = 256
        block7 = _conv_bn_relu_x2(nb_filter, 3, 3)(up6)     # filters: 512 -> 256
        up_conv7 = UpSampling2D(size=(2, 2))(block7)        # filters: 256 -> 128
        up7 = Concatenate([block2, up_conv7], axis=-1)      # concat: 128 + 128 = 256
        # =========================================================================
        nb_filter /= 2  # default nb_filter = 128
        block8 = _conv_bn_relu_x2(nb_filter, 3, 3)(up7)     # filters: 256 -> 128
        up_conv8 = UpSampling2D(size=(2, 2))(block8)        # filters: 128 -> 64
        up8 = Concatenate([block1, up_conv8], axis=-1)      # concat: 64 + 64 -> 128
        # =========================================================================
        nb_filter /= 2  # default nb_filter = 64
        act_ = _conv_bn_relu_x2(nb_filter, 3, 3)(up8)       # filters: 128 -> 64
        # =========================================================================
        density_pred = Convolution2D(1, 1, 1, bias=False, activation='linear',
                                     init='orthogonal', name='pred', border_mode='same')(act_)
        # =========================================================================
        model = Model(input=input_, output=density_pred)
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        return model

    def train(self):
        print("loading data")
        (x_train, y_train), (x_val, y_val) = load_data()
        print("loading data done")
        model = self.buildModel_U_net()
        print("got U-net")

        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(x_train, y_train, batch_size=1, nb_epoch=10, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

        print('predict test data')
        y_val = model.predict(x_val, batch_size=1, verbose=1)
        np.save('y_val.npy', y_val)


if __name__ == '__main__':
    eggstimator = EggCountNet()
    eggstimator.train()

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: Weidi Xie

Adapted by Kevin Yost

@description:
This is the file to create the model, similar as the paper, but with batch normalization, make it more easier to train.

TODO:   Feed data in properly. Fix Concatenation.
"""

import numpy as np
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Convolution2D, UpSampling2D, Concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from data import load_train_data, load_test_data, image_crop_and_scale


class EggCountNet(object):
    def __init__(self, img_rows=480, img_cols=640):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def buildModel_U_net(self):
        input_ = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(input_)
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool1)
        block2 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
        # =========================================================================
        block3 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool2)
        block3 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
        # =========================================================================
        block4 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool3)
        block4 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(block4)
        # =========================================================================
        block5 = Convolution2D(filters=1024, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool4)
        block5 = Convolution2D(filters=1024, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block5)
        up_conv5 = UpSampling2D(size=(2, 2))(block5)    # filters: 1024 -> 512
        up5 = Concatenate([block4, up_conv5])           # concat: 512 + 512 = 1024
        # =========================================================================
        block6 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up5)
        block6 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block6)
        up_conv6 = UpSampling2D(size=(2, 2))(block6)    # filters: 512 -> 256
        up6 = Concatenate([block3, up_conv6])           # concat: 256 + 256 = 512
        # =========================================================================
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up6)
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block7)
        up_conv7 = UpSampling2D(size=(2, 2))(block7)    # filters: 256 -> 128
        up7 = Concatenate([block2, up_conv7])           # concat: 128 + 128 = 256
        # =========================================================================
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up7)
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block8)
        up_conv8 = UpSampling2D(size=(2, 2))(block8)    # filters: 128 -> 64
        up8 = Concatenate([block1, up_conv8])           # concat: 64 + 64 -> 128
        # =========================================================================
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up8)
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block9)
        # =========================================================================
        density_pred = Convolution2D(1, 1, 1, bias=False, activation='linear',
                                     init='orthogonal', name='pred', border_mode='same')(block9)
        # =========================================================================
        model = Model(input=input_, output=density_pred)
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        return model

    def train(self):
        print("loading data")
        X_train, Y_train = load_train_data()
        X_test = load_test_data()
        print("loading data done")
        model = self.buildModel_U_net()
        print("got U-net")

        model_checkpoint = ModelCheckpoint('eggstimator.hdf5', monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')
        model.fit(X_train, Y_train, batch_size=10, validation_split=0.2, nb_epoch=10, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint])

        print('predict test data')
        Y_val = model.predict(X_test, batch_size=1, verbose=1)
        np.save('Y_val.npy', Y_val)


if __name__ == '__main__':
    eggstimator = EggCountNet()
    eggstimator.train()

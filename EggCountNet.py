# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: Kevin Yost

@description:

TODO:
    - add more data - random section cropping? regularize, batch norm
    - load model properly in predict - get last loss
"""

import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from data import load_train_data, load_test_data, image_crop_and_scale, average_egg_density_coefficient


class EggCountNet(object):
    def __init__(self, img_rows=480, img_cols=640):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def buildModel_U_net_9block(self, pre_trained_weights: str = None):
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
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
        drop4 = Dropout(0.5)(block4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        # =========================================================================
        block5 = Convolution2D(filters=1024, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool4)
        block5 = Convolution2D(filters=1024, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block5)
        drop5 = Dropout(0.5)(block5)
        up_samp5 = UpSampling2D(size=(2, 2))(drop5)
        up_conv5 = Convolution2D(filters=512, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp5)
        up5 = concatenate([drop4, up_conv5])  # concat: 512 + 512 = 1024
        # =========================================================================
        block6 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up5)
        block6 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block6)
        up_samp6 = UpSampling2D(size=(2, 2))(block6)
        up_conv6 = Convolution2D(filters=256, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp6)
        up6 = concatenate([block3, up_conv6])  # concat: 256 + 256 = 512
        # =========================================================================
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up6)
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block7)
        up_samp7 = UpSampling2D(size=(2, 2))(block7)
        up_conv7 = Convolution2D(filters=128, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp7)
        up7 = concatenate([block2, up_conv7])  # concat: 128 + 128 = 256
        # =========================================================================
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up7)
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block8)
        up_samp8 = UpSampling2D(size=(2, 2))(block8)
        up_conv8 = Convolution2D(filters=64, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp8)
        up8 = concatenate([block1, up_conv8])  # concat: 64 + 64 -> 128
        # =========================================================================
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up8)
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block9)  # (480, 640, 64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=True)(block9)
        # =========================================================================
        model = Model(inputs=inputs, outputs=density_pred)
        optim = Adam()
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def buildModel_U_net_7block(self, pre_trained_weights: str = None):
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
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
        up_samp4 = UpSampling2D(size=(2, 2))(block4)
        up_conv4 = Convolution2D(filters=256, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp4)
        up4 = concatenate([block3, up_conv4])  # concat: 256+256=512
        # =========================================================================
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up4)
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block7)
        up_samp7 = UpSampling2D(size=(2, 2))(block7)
        up_conv7 = Convolution2D(filters=128, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp7)
        up7 = concatenate([block2, up_conv7])  # concat: 128 + 128 = 256
        # =========================================================================
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up7)
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block8)
        up_samp8 = UpSampling2D(size=(2, 2))(block8)
        up_conv8 = Convolution2D(filters=64, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp8)
        up8 = concatenate([block1, up_conv8])  # concat: 64 + 64 -> 128
        # =========================================================================
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up8)
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block9)  # (480, 640, 64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=True)(block9)
        # =========================================================================
        model = Model(inputs=inputs, outputs=density_pred)
        optim = Adam()
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def buildModel_U_net_5block(self, pre_trained_weights: str = None):
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool1)
        block2 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block2)
        drop2 = Dropout(0.5)(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        # =========================================================================
        block3 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool2)
        block3 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block3)
        drop3 = Dropout(0.5)(block3)
        up_samp3 = UpSampling2D(size=(2, 2))(drop3)
        up_conv3 = Convolution2D(filters=128, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp3)
        up3 = concatenate([drop2, up_conv3])  # concat: 128+128=256
        # =========================================================================
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up3)
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block8)
        up_samp8 = UpSampling2D(size=(2, 2))(block8)
        up_conv8 = Convolution2D(filters=64, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp8)
        up8 = concatenate([block1, up_conv8])  # concat: 64 + 64 -> 128
        # =========================================================================
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up8)
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block9)  # (480, 640, 64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=True)(block9)
        # =========================================================================
        model = Model(inputs=inputs, outputs=density_pred)
        optim = Adam()
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def train(self, model_file):
        print("Loading data...")
        X_train, Y_train = load_train_data()
        print("Loading data done")

        if exists(model_file):
            model = load_model(model_file)
        else:
            if model_file == 'model_unet5.h5':
                model = self.buildModel_U_net_5block()
            elif model_file == 'model_unet7.h5':
                model = self.buildModel_U_net_7block()
            elif model_file == 'model_unet9.h5':
                model = self.buildModel_U_net_9block()

        print("Got U-net:\t" + model_file)

        # model.summary()

        def scheduler(epoch):
            lr_init = 1e-8
            ep_switch = 5
            if epoch < ep_switch:
                return lr_init
            else:
                return lr_init * np.exp(0.05 * (ep_switch - epoch))

        model_learning_rate = LearningRateScheduler(scheduler, verbose=1)
        model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')

        # batch_size limited to 1 due to my own GPU's memory limitations
        history = model.fit(X_train, Y_train, batch_size=1, validation_split=0.2, epochs=50, verbose=1,
                            shuffle=True, callbacks=[model_checkpoint, model_learning_rate])

        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        # print('predict test data')
        # X_test = load_test_data()
        # Y_test = model.predict(X_test, batch_size=1, verbose=1)
        # np.save('Y_test.npy', Y_test)

    def predict(self, model_file):
        if exists(model_file):
            model = load_model(model_file)
        else:
            print("Model doesn't exist. Nothing to validate.")
            exit()
        img = image_crop_and_scale()
        img = img[np.newaxis, ...]
        y_pred = model.predict(img)
        heat_pred = np.sum(y_pred)
        print('Heat predicted:\t' + str(heat_pred))
        eggs_pred = int(np.round(heat_pred / average_egg_density_coefficient()))
        print('Predicted number of eggs:\t' + str(eggs_pred))

    def validate(self, model_file):
        if exists(model_file):
            model = load_model(model_file)
        else:
            print("Model doesn't exist. Nothing to validate.")
            exit()
        X_train, Y_train = load_train_data()
        Y_pred = model.predict(X_train, batch_size=1, verbose=1)

        for i in range(Y_pred.shape[0]):
            heat_pred = np.sum(Y_pred[i])
            print('Heat predicted:\t' + str(heat_pred))
            eggs_pred = int(np.round(heat_pred / 100))
            print('Predicted number of eggs:\t' + str(eggs_pred))
            ground_truth = int(np.round(np.sum(Y_train[i]) / 100))
            print('Actual number of eggs:\t' + str(ground_truth))
            error = eggs_pred - ground_truth
            print('Error:\t' + str(error))
            print('-' * 30)


if __name__ == '__main__':
    eggstimator = EggCountNet()
    model_file = 'model_unet9.h5'
    eggstimator.train(model_file)
    # eggstimator.predict(model_file)
    # eggstimator.validate(model_file)

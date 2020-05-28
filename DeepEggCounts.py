# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 19:13:36 2017

@author: Kevin Yost

@description:


"""
import datetime
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from data import *


def conv_bn_relu_x2(input_layer, filters):
    """
    Applies Conv2D, Batch normalization, and RELU activation twice to input layer
    Args:
        input_layer:    input layer
        filters:        number of filters in conv2d layers

    Returns:
        block:          output after being processed
    """
    block = Convolution2D(filters=filters, kernel_size=3, use_bias=False, padding='same',
                          kernel_initializer='he_normal')(input_layer)
    block = BatchNormalization()(block)
    block = Activation(activation='relu')(block)
    block = Convolution2D(filters=filters, kernel_size=3, use_bias=False, padding='same',
                          kernel_initializer='he_normal')(block)
    block = BatchNormalization()(block)
    block = Activation(activation='relu')(block)
    return block


def up_conv_bn_relu_cat(input_layer, concat_layer, filters):
    """
    Up-samples input layer, applies conv2d, batch norm., and RELU activation.
    Args:
        input_layer:
        concat_layer:
        filters:

    Returns:

    """
    block = UpSampling2D(size=(2, 2))(input_layer)
    # ensuring valid concat
    if block.shape[1] != concat_layer.shape[1]:
        block = ZeroPadding2D(((0, 1), (0, 0)))(block)
    if block.shape[2] != concat_layer.shape[2]:
        block = ZeroPadding2D(((0, 0), (0, 1)))(block)
    block = Convolution2D(filters=filters, kernel_size=3, use_bias=False, padding='same',
                          kernel_initializer='he_normal')(block)
    block = BatchNormalization()(block)
    block = Activation(activation='relu')(block)
    block = concatenate([concat_layer, block])
    return block


def validate_test(model_file: str = 'model_u_net_9.h5'):
    if exists(model_file):
        model = load_model(model_file)
    else:
        print("Model doesn't exist. Nothing to validate.")
        exit()

    print('-' * 100)
    print('Train and Validation set:')
    X, Y, n_sub_imgs = load_data_npy()
    # s = np.random.randint(X.shape[0])
    s = 1780
    X = X[s:s + 20]
    Y = Y[s:s + 20]
    Y_pred = model.predict(X, batch_size=1, verbose=1)

    # n_pred = sum_density_over_sub_images(Y_pred, n_sub_imgs)
    # n_actual = sum_density_over_sub_images(Y, n_sub_imgs)
    # for i in range(n_pred.shape[0]):
    #     print('Predicted number of eggs:\t' + str(n_pred[i]))
    #     print('Actual number of eggs:\t' + str(n_actual[i]))
    #     error = n_pred[i] - n_actual[i]
    #     print('Error:\t' + str(error) + ' (' + f'{catch_div_by_zero(error, n_actual[i])*100:.2f}' + '%)')
    #     print('-' * 30)

    for i in range(Y_pred.shape[0]):
        a = np.sum(Y[i]) / 100
        p = np.sum(Y_pred[i]) / 100
        print('Predicted number of eggs:\t' + str(p))
        print('Actual number of eggs:\t' + str(a))
        error = p - a
        # print('Error:\t' + str(error) + ' (' + f'{catch_div_by_zero(error, a)*100:.2f}' + '%)')
        print('Error:\t' + str(error) + ' (' + f'{(0 if a == 0 else error / a) * 100:.2f}' + '%)')
        print('-' * 30)

    for i in range(5):
        rn = np.random.randint(Y_pred.shape[0])
        heat_map(X[rn], Y[rn], Y_pred[rn])


def predict_input(model_file: str = 'model_u_net_9.h5'):
    if exists(model_file):
        model = load_model(model_file)
    else:
        print("Model doesn't exist. Nothing to validate.")
        exit()
    img = image_crop_and_scale()
    img = img[np.newaxis, ...]      # add axis for single image inputs
    y_pred = model.predict(img)
    heat_pred = np.sum(y_pred[0])
    egg_density = 100
    print('Heat predicted:\t' + str(heat_pred))
    eggs_pred = int(np.round(heat_pred / egg_density))
    print('Predicted number of eggs:\t' + str(eggs_pred))
    print('-' * 30)


class DeepEggCounts(object):
    def __init__(self, img_rows=256, img_cols=256):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def build_u_net_9(self, pre_trained_weights: str = None):
        # experimenting with different U-net depths
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = conv_bn_relu_x2(input_layer=inputs, filters=64)
        drop1 = Dropout(0.25)(block1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = conv_bn_relu_x2(input_layer=pool1, filters=128)
        drop2 = Dropout(0.1)(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        # =========================================================================
        block3 = conv_bn_relu_x2(input_layer=pool2, filters=256)
        drop3 = Dropout(0.1)(block3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        # =========================================================================
        block4 = conv_bn_relu_x2(input_layer=pool3, filters=512)
        drop4 = Dropout(0.1)(block4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        # =========================================================================
        block5 = conv_bn_relu_x2(input_layer=pool4, filters=1024)
        drop5 = Dropout(0.1)(block5)
        up5 = up_conv_bn_relu_cat(input_layer=drop5, concat_layer=drop4, filters=512)
        # =========================================================================
        block6 = conv_bn_relu_x2(input_layer=up5, filters=512)
        up6 = up_conv_bn_relu_cat(input_layer=block6, concat_layer=drop3, filters=256)
        # =========================================================================
        block7 = conv_bn_relu_x2(input_layer=up6, filters=256)
        up7 = up_conv_bn_relu_cat(input_layer=block7, concat_layer=drop2, filters=128)
        # =========================================================================
        block8 = conv_bn_relu_x2(input_layer=up7, filters=128)
        up8 = up_conv_bn_relu_cat(input_layer=block8, concat_layer=drop1, filters=64)
        # =========================================================================
        block9 = conv_bn_relu_x2(input_layer=up8, filters=64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=False)(block9)
        # =========================================================================
        model = Model(inputs=inputs, outputs=density_pred)
        optim = Adam(lr=1e-4)
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def build_u_net_7(self, pre_trained_weights: str = None):
        # experimenting with different U-net depths
        inputs = Input((self.img_rows, self.img_cols, 3))
        drop_in = Dropout(0.1)(inputs)
        # =========================================================================
        block1 = conv_bn_relu_x2(input_layer=drop_in, filters=64)
        drop1 = Dropout(0.3)(block1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
        # =========================================================================
        block2 = conv_bn_relu_x2(input_layer=pool1, filters=128)
        drop2 = Dropout(0.3)(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        # =========================================================================
        block3 = conv_bn_relu_x2(input_layer=pool2, filters=256)
        drop3 = Dropout(0.2)(block3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        # =========================================================================
        block4 = conv_bn_relu_x2(input_layer=pool3, filters=512)
        drop4 = Dropout(0.2)(block4)
        up4 = up_conv_bn_relu_cat(input_layer=drop4, concat_layer=drop3, filters=256)
        # =========================================================================
        block5 = conv_bn_relu_x2(input_layer=up4, filters=256)
        drop5 = Dropout(0.1)(block5)
        up5 = up_conv_bn_relu_cat(input_layer=drop5, concat_layer=drop2, filters=128)
        # =========================================================================
        block6 = conv_bn_relu_x2(input_layer=up5, filters=128)
        drop6 = Dropout(0.1)(block6)
        up6 = up_conv_bn_relu_cat(input_layer=drop6, concat_layer=drop1, filters=64)
        # =========================================================================
        block7 = conv_bn_relu_x2(input_layer=up6, filters=64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=False)(block7)
        # =========================================================================
        model = Model(inputs=inputs, outputs=density_pred)
        optim = SGD(lr=1e-4, momentum=0.9)
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def build_u_net_5(self, pre_trained_weights: str = None):
        # experimenting with different U-net depths
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = conv_bn_relu_x2(input_layer=inputs, filters=64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = conv_bn_relu_x2(input_layer=pool1, filters=128)
        drop2 = Dropout(0.5)(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        # =========================================================================
        block3 = conv_bn_relu_x2(input_layer=pool2, filters=256)
        block3 = Dropout(0.5)(block3)
        up3 = up_conv_bn_relu_cat(block3, drop2, 128)  # concat: 128+128=256
        # =========================================================================
        block4 = conv_bn_relu_x2(input_layer=up3, filters=128)
        up4 = up_conv_bn_relu_cat(block4, block1, 64)
        # =========================================================================
        block5 = conv_bn_relu_x2(input_layer=up4, filters=64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=False)(block5)
        # =========================================================================
        model = Model(inputs=inputs, outputs=density_pred)
        optim = SGD(lr=1e-4, momentum=0.9)
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def build_heat_summary(self):
        inputs = Input((self.img_rows, self.img_cols, 1))
        block1 = conv_bn_relu_x2(input_layer=inputs, filters=1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        flat1 = Flatten()(pool1)
        dense1 = Dense(128, activation='relu')(flat1)
        output = Dense(1, activation='relu')(dense1)
        model = Model(inputs=inputs, outputs=output)
        optim = Adam(lr=1e-4)
        model.compile(optimizer=optim, loss='mse', metrics=['acc'])
        return model


    def train(self, model_file: str = ''):
        print("Loading data...")
        if 'e2e' in model_file:
            X, Y, n_sub_imgs = load_data_npy(e2e=True)
        else:
            X, Y, n_sub_imgs = load_data_npy()
        # s = np.random.randint(X.shape[0])
        # s = 1780
        # n = 50
        # X = X[s:s+n]
        # Y = Y[s:s+n]

        print("Loading data done")

        if exists(model_file):
            model = load_model(model_file)
            print("Got U-net:\t" + model_file)
        else:
            if model_file == 'model_u_net_5.h5':
                model = self.build_u_net_5()
            elif model_file == 'model_u_net_7.h5':
                model = self.build_u_net_7()
            elif model_file == 'model_u_net_9.h5':
                model = self.build_u_net_9()
            elif model_file == 'model_u_net_5_e2e.h5':
                model = self.build_u_net_5_e2e()
            print("Built U-net:\t" + model_file)

        # model.summary()

        # Callbacks
        def scheduler(epoch):
            return 10e-4 * np.power(0.975, epoch)     # 2.5% LR drop per epoch
        model_learning_rate = LearningRateScheduler(scheduler, verbose=1)
        model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True)

        # batch_size limited to 1 due to my own GPU's memory limitations
        print('Fitting model...')
        history = model.fit(X, Y, batch_size=1, validation_split=0.1, shuffle=True,
                            initial_epoch=0, epochs=200,
                            # steps_per_epoch=500, validation_steps=100,
                            verbose=1, callbacks=[model_checkpoint])

        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        validate_test(model_file)


if __name__ == '__main__':
    eggstimator = DeepEggCounts()
    model_file = 'model_u_net_9.h5'
    eggstimator.train(model_file)
    # validate_test(model_file)
    # predict_input(model_file)

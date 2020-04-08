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


# TODO:
#     - load model properly in predict - get last loss
#     - envelope network with UI

def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
        number_size = 2.0
    if K.floatx() == 'float64':
        number_size = 8.0

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3) + internal_model_mem_count
    return gbytes


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


def validate_test(model_file: str = 'model_unet9.h5'):
    if exists(model_file):
        model = load_model(model_file)
    else:
        print("Model doesn't exist. Nothing to validate.")
        exit()
    egg_density = 100

    print('-' * 100)
    print('Train and Validation set:')
    X_train, Y_train, X_val, Y_val = load_train_val_data()
    Y_pred_train = model.predict(X_train, batch_size=1, verbose=1)
    Y_pred_val = model.predict(X_val, batch_size=1, verbose=1)
    err_train = 0
    err_val = 0
    for i in range(Y_pred_train.shape[0]):
        heat_pred = np.sum(Y_pred_train[i])
        eggs_pred = int(np.round(heat_pred / egg_density))
        print('Predicted number of eggs:\t' + str(eggs_pred))
        ground_truth = int(np.round(np.sum(Y_train[i]) / egg_density))
        print('Actual number of eggs:\t' + str(ground_truth))
        error = eggs_pred - ground_truth
        # heat_map(X_train[i], Y_pred[i])
        print('Error:\t' + str(error) + ' (' + f'{catch_div_by_zero(error, ground_truth)*100:.2f}' + '%)')
        err_train += np.abs(error)
        print('-' * 30)
    print('Total error in train set:\t' + str(err_train))

    print('-' * 100)
    for i in range(Y_pred_val.shape[0]):
        heat_pred = np.sum(Y_pred_val[i])
        eggs_pred = int(np.round(heat_pred / egg_density))
        print('Predicted number of eggs:\t' + str(eggs_pred))
        ground_truth = int(np.round(np.sum(Y_val[i]) / egg_density))
        print('Actual number of eggs:\t' + str(ground_truth))
        error = eggs_pred - ground_truth
        heat_map(X_val[i], Y_pred_val[i])
        print('Error:\t' + str(error) + ' (' + f'{catch_div_by_zero(error, ground_truth)*100:.2f}' + '%)')
        err_val += np.abs(error)
        print('-' * 30)
    print('Total error in validation set:\t' + str(err_val))


    # print('-' * 100)
    # print('Test set')
    # X_test = load_test_data()
    # Y_test = model.predict(X_test, batch_size=1, verbose=1)
    # for i in range(Y_test.shape[0]):
    #     heat_pred = np.sum(Y_test[i])
    #     eggs_pred = int(np.round(heat_pred / egg_density))
    #     print('Predicted number of eggs:\t' + str(eggs_pred))
    #     # ground_truth = int(np.round(np.sum(Y_test[i]) / egg_density))
    #     # print('Actual number of eggs:\t' + str(ground_truth))
    #     # error = eggs_pred - ground_truth
    #     heat_map(X_test[i], Y_test[i])
    #     # print('Error:\t' + str(error) + ' (' + f'{catch_div_by_zero(error, ground_truth)*100:.2f}' + '%)')
    #     # err_val += np.abs(error)
    #     print('-' * 30)


def predict_input(model_file: str = 'model_unet9.h5'):
    if exists(model_file):
        model = load_model(model_file)
    else:
        print("Model doesn't exist. Nothing to validate.")
        exit()
    img = image_crop_and_scale()
    img = img[np.newaxis, ...]
    y_pred = model.predict(img)
    heat_pred = np.sum(y_pred[i])
    egg_density = 100
    print('Heat predicted:\t' + str(heat_pred))
    eggs_pred = int(np.round(heat_pred / egg_density))
    print('Predicted number of eggs:\t' + str(eggs_pred))
    print('-' * 30)


def catch_div_by_zero(n, d):
    return 0 if d == 0 else n / d

class EggCountNet(object):
    def __init__(self, img_rows=360, img_cols=480):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def buildModel_U_net_9block(self, pre_trained_weights: str = None):
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = conv_bn_relu_x2(input_layer=inputs, filters=64)
        drop1 = Dropout(0.0)(block1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = conv_bn_relu_x2(input_layer=pool1, filters=128)
        drop2 = Dropout(0.5)(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        # =========================================================================
        block3 = conv_bn_relu_x2(input_layer=pool2, filters=256)
        drop3 = Dropout(0.3)(block3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        # =========================================================================
        block4 = conv_bn_relu_x2(input_layer=pool3, filters=512)
        drop4 = Dropout(0.3)(block4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        # =========================================================================
        block5 = conv_bn_relu_x2(input_layer=pool4, filters=1024)
        up5 = up_conv_bn_relu_cat(input_layer=block5, concat_layer=drop4, filters=512)
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
        optim = Adam()
        model.compile(optimizer=optim, loss='mean_squared_error')
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def buildModel_U_net_7block(self, pre_trained_weights: str = None):
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
        pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
        # =========================================================================
        block4 = conv_bn_relu_x2(input_layer=pool3, filters=512)
        up4 = up_conv_bn_relu_cat(input_layer=block4, concat_layer=block3, filters=256)
        # =========================================================================
        block7 = conv_bn_relu_x2(input_layer=up4, filters=256)
        drop7 = Dropout(0.5)(block7)
        up7 = up_conv_bn_relu_cat(input_layer=drop7, concat_layer=block2, filters=128)
        # =========================================================================
        block8 = conv_bn_relu_x2(input_layer=up7, filters=128)
        up8 = up_conv_bn_relu_cat(input_layer=block8, concat_layer=block1, filters=64)
        # =========================================================================
        block9 = conv_bn_relu_x2(input_layer=up8, filters=64)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=False)(block9)
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
        optim = Adam()
        model.compile(optimizer=optim, loss='mean_squared_error', metrics=['acc'])
        if pre_trained_weights:
            model.load_weights(pre_trained_weights)
            print('weights loaded')
        return model

    def train(self, model_file):
        print("Loading data...")
        X_train, Y_train, X_val, Y_val = load_train_val_data()

        print("Loading data done")

        if exists(model_file):
            model = load_model(model_file)
            print("Got U-net:\t" + model_file)
        else:
            if model_file == 'model_unet5.h5':
                model = self.buildModel_U_net_5block()
            elif model_file == 'model_unet7.h5':
                model = self.buildModel_U_net_7block()
            elif model_file == 'model_unet9.h5':
                model = self.buildModel_U_net_9block()
            print("Built U-net:\t" + model_file)

        model.summary()
        # get_model_memory_usage(1, model)

        def scheduler(epoch):
            lr_init = 1e-4
            ep_switch = 10
            if epoch < ep_switch:
                return lr_init
            else:
                return lr_init * np.power(0.95, epoch - ep_switch)

        model_learning_rate = LearningRateScheduler(scheduler, verbose=1)
        model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')

        # batch_size limited to 1 due to my own GPU's memory limitations
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(X_train, Y_train, batch_size=1, validation_data=(X_val, Y_val), epochs=200, verbose=1,
                            shuffle=True, callbacks=[model_checkpoint, model_learning_rate])

        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        validate_test(model_file)


if __name__ == '__main__':
    eggstimator = EggCountNet()
    model_file = 'model_unet9.h5'
    # eggstimator.train(model_file)
    validate_test(model_file)
    # predict_input(model_file)

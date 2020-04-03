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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import *
from data import *


# TODO:
#     - solve OOM: assign weights to CPU?
#     - load model properly in predict - get last loss
#     -

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

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
    block = UpSampling2D(size=(2, 2))(input_layer)
    block = Convolution2D(filters=filters, kernel_size=3, use_bias=False, padding='same',
                          kernel_initializer='he_normal')(block)
    block = BatchNormalization()(block)
    block = Activation(activation='relu')(block)
    block = concatenate([concat_layer, block])
    return block


class EggCountNet(object):
    def __init__(self, img_rows=360, img_cols=480):
        self.img_rows = img_rows
        self.img_cols = img_cols

    def buildModel_U_net_9block(self, pre_trained_weights: str = None):
        drop_out_factor = 0.5
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(inputs)
        block1 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block1)
        block1 = BatchNormalization()(block1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool1)
        block2 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block2)
        block2 = BatchNormalization()(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
        # =========================================================================
        block3 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool2)
        block3 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block3)
        block3 = BatchNormalization()(block3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
        # =========================================================================
        block4 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool3)
        block4 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block4)
        block4 = BatchNormalization()(block4)
        drop4 = Dropout(drop_out_factor)(block4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        # =========================================================================
        block5 = Convolution2D(filters=1024, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(pool4)
        block5 = Convolution2D(filters=1024, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block5)
        block5 = BatchNormalization()(block5)
        drop5 = Dropout(drop_out_factor)(block5)
        up_samp5 = UpSampling2D(size=(2, 2))(drop5)
        up_conv5 = Convolution2D(filters=512, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp5)
        # up_conv5 = BatchNormalization()(up_conv5)
        up5 = concatenate([drop4, up_conv5])  # concat: 512 + 512 = 1024
        # =========================================================================
        block6 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up5)
        block6 = Convolution2D(filters=512, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block6)
        block6 = BatchNormalization()(block6)
        up_samp6 = UpSampling2D(size=(2, 2))(block6)
        up_conv6 = Convolution2D(filters=256, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp6)
        # up_conv6 = BatchNormalization()(up_conv6)
        up6 = concatenate([block3, up_conv6])  # concat: 256 + 256 = 512
        # =========================================================================
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up6)
        block7 = Convolution2D(filters=256, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block7)
        block7 = BatchNormalization()(block7)
        up_samp7 = UpSampling2D(size=(2, 2))(block7)
        up_conv7 = Convolution2D(filters=128, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp7)
        # up_conv7 = BatchNormalization()(up_conv7)
        up7 = concatenate([block2, up_conv7])  # concat: 128 + 128 = 256
        # =========================================================================
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up7)
        block8 = Convolution2D(filters=128, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block8)
        block8 = BatchNormalization()(block8)
        up_samp8 = UpSampling2D(size=(2, 2))(block8)
        up_conv8 = Convolution2D(filters=64, kernel_size=2, activation='relu', padding='same',
                                 kernel_initializer='he_normal')(up_samp8)
        # up_conv8 = BatchNormalization()(up_conv8)
        up8 = concatenate([block1, up_conv8])  # concat: 64 + 64 -> 128
        # =========================================================================
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(up8)
        block9 = Convolution2D(filters=64, kernel_size=3, activation='relu', padding='same',
                               kernel_initializer='he_normal')(block9)  # (480, 640, 64)
        block9 = BatchNormalization()(block9)
        # =========================================================================
        density_pred = Convolution2D(filters=1, kernel_size=1, activation='relu', padding='same',
                                     kernel_initializer='he_normal', use_bias=True)(block9)
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
        pool2 = MaxPooling2D(pool_size=(2, 2))(block2)
        # =========================================================================
        block3 = conv_bn_relu_x2(input_layer=pool2, filters=256)
        pool3 = MaxPooling2D(pool_size=(2, 2))(block3)
        # =========================================================================
        block4 = conv_bn_relu_x2(input_layer=pool3, filters=512)
        up4 = up_conv_bn_relu_cat(input_layer=block4, concat_layer=block3, filters=256)
        # =========================================================================
        block7 = conv_bn_relu_x2(input_layer=up4, filters=256)
        up7 = up_conv_bn_relu_cat(input_layer=block7, concat_layer=block2, filters=128)
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
        drop_out_factor = 0.5
        inputs = Input((self.img_rows, self.img_cols, 3))
        # =========================================================================
        block1 = conv_bn_relu_x2(input_layer=inputs, filters=64)
        pool1 = MaxPooling2D(pool_size=(2, 2))(block1)
        # =========================================================================
        block2 = conv_bn_relu_x2(input_layer=pool1, filters=128)
        drop2 = Dropout(drop_out_factor)(block2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        # =========================================================================
        block3 = conv_bn_relu_x2(input_layer=pool2, filters=256)
        block3 = Dropout(drop_out_factor)(block3)
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
        X_train, Y_train = load_train_data()

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
            # return lr_init / (np.power(10, epoch/2))
            if epoch < ep_switch:
                return lr_init
            else:
                return lr_init * np.power(0.9, epoch - ep_switch)

        model_learning_rate = LearningRateScheduler(scheduler, verbose=1)
        model_checkpoint = ModelCheckpoint(model_file, monitor='loss', verbose=1, save_best_only=True)
        print('Fitting model...')

        # batch_size limited to 1 due to my own GPU's memory limitations
        # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        history = model.fit(X_train, Y_train, batch_size=1, validation_split=0.2, epochs=20, verbose=1,
                            shuffle=True, callbacks=[model_checkpoint, model_learning_rate])

        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()

        self.validate(model_file)

    def predict(self, model_file):
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

    def validate(self, model_file):
        if exists(model_file):
            model = load_model(model_file)
        else:
            print("Model doesn't exist. Nothing to validate.")
            exit()
        X_train, Y_train = load_train_data()
        Y_pred = model.predict(X_train, batch_size=1, verbose=1)
        egg_density = 100

        for i in range(Y_pred.shape[0]):
            heat_pred = np.sum(Y_pred[i])
            print('Heat predicted:\t' + str(heat_pred))
            eggs_pred = int(np.round(heat_pred / egg_density))
            print('Predicted number of eggs:\t' + str(eggs_pred))
            ground_truth = int(np.round(np.sum(Y_train[i]) / egg_density))
            print('Actual number of eggs:\t' + str(ground_truth))
            error = eggs_pred - ground_truth
            heat_map(X_train[i], Y_pred[i])
            print('Error:\t' + str(error))
            print('-' * 30)


if __name__ == '__main__':
    eggstimator = EggCountNet()
    model_file = 'model_unet7.h5'
    eggstimator.train(model_file)
    # eggstimator.validate(model_file)
    # eggstimator.predict(model_file)

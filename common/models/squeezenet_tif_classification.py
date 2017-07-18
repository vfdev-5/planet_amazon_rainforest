
import os
import sys

from keras.optimizers import Adadelta, Adam, SGD, Nadam
from keras.layers import Flatten, Dense
from .keras_metrics import precision, recall
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, BatchNormalization
from keras.layers.convolutional import SeparableConv2D
from keras.models import Model


sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"
bn = "bn_"


def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, (1, 1), padding='valid', use_bias=False, name=s_id + sq1x1)(x)
    x = BatchNormalization(name=s_id + bn + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution2D(expand, (1, 1), padding='valid', use_bias=False, name=s_id + exp1x1)(x)
    left = BatchNormalization(name=s_id + bn + exp1x1)(left)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution2D(expand, (3, 3), padding='same', use_bias=False, name=s_id + exp3x3)(x)
    right = BatchNormalization(name=s_id + bn + exp3x3)(right)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x


def get_squeezenet_on_tif(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    final_activation = 'sigmoid'

    inputs = Input(input_shape)

    x = SeparableConv2D(64, (3, 3), strides=(2, 2), use_bias=False, padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)

    name = 'conv10_%i' % n_classes
    x = Convolution2D(n_classes, (1, 1), padding='valid', name=name)(x)
    x = Activation('relu', name='relu_conv10')(x)

    x = Flatten()(x)
    x = Dense(96, activation='relu', name='d1')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, name='d2')(x)

    outputs = Activation(final_activation, name='tag_vector')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.name = "Separable_SqueezeNet_BN_on_tif"

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'nadam':
        opt = Nadam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss=loss, optimizer=opt, metrics=[precision, recall])
    return model
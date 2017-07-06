
import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
squeezenet_path = os.path.abspath(os.path.join(current_path, 'KerasSqueezeNet', 'keras_squeezenet'))
if squeezenet_path not in sys.path:
    sys.path.append(squeezenet_path)

from squeezenet import SqueezeNet
from keras.optimizers import Adadelta, Adam, SGD
from keras.layers import GlobalAveragePooling2D, Activation, Flatten, Dense, Dropout
from keras.models import Model
from .keras_metrics import precision, recall


def get_squeezenet(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    weights = 'imagenet' if 'weights' not in params else params['weights']
    final_activation = 'sigmoid'

    snet = SqueezeNet(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights)

    x = snet.outputs[0]
    x = GlobalAveragePooling2D()(x)

    out = Activation(final_activation, name='tag_vector')(x)
    model = Model(inputs=snet.inputs, outputs=out)

    model.name = "SqueezeNet"

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss=loss, optimizer=opt, metrics=[precision, recall])
    return model


def get_squeezenet2(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    weights = 'imagenet' if 'weights' not in params else params['weights']

    snet = SqueezeNet(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights)

    x = snet.outputs[0]
    x = Flatten()(x)
    x = Dense(64, activation='relu', name='d1')(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, name='d2')(x)
    out = Activation('sigmoid', name='tags')(x)
    model = Model(inputs=snet.inputs, outputs=out)

    model.name = "SqueezeNet2"

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss=loss, optimizer=opt, metrics=[precision, recall])
    return model


import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
densenet_path = os.path.abspath(os.path.join(current_path, 'KerasDenseNet'))
if densenet_path not in sys.path:
    sys.path.append(densenet_path)

from keras.optimizers import Adadelta, Adam, SGD
from keras.layers import GlobalAveragePooling2D, Activation, Flatten, Dense
from keras.models import Model
from .keras_metrics import precision, recall
from .densenet import DenseNet


def get_densenet(depth, input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    weights = 'imagenet' if 'weights' not in params else params['weights']
    final_activation = 'sigmoid'

    dnet = DenseNet(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights, depth=depth)

    x = dnet.outputs[0]
    x = GlobalAveragePooling2D()(x)

    out = Activation(final_activation, name='loss')(x)
    model = Model(inputs=dnet.inputs, outputs=out)

    # Set some layers trainable
    # for layer in model.layers:
    #     if layer.name in names_to_train:
    #         layer.trainable = True
    #     else:
    #         layer.trainable = False

    model.name = "DenseNet"

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


import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
squeezenet_path = os.path.abspath(os.path.join(current_path, 'KerasSqueezeNet', 'keras_squeezenet'))
if squeezenet_path not in sys.path:
    sys.path.append(squeezenet_path)

from squeezenet import SqueezeNet
from keras.optimizers import Adadelta, Adam, SGD
from keras.layers import GlobalAveragePooling2D, Activation
from keras.models import Model
from .keras_metrics import precision, recall


def get_squeezenet(input_shape, n_classes, optimizer='', lr=0.01, loss='', weights='imagenet'):
    """
    """

    names_to_train = [
        'fire5/squeeze1x1', 'fire5/expand1x1', 'fire5/expand3x3',
        'fire6/squeeze1x1', 'fire6/expand1x1', 'fire6/expand3x3',
        'fire7/squeeze1x1', 'fire7/expand1x1', 'fire7/expand3x3',
        'fire8/squeeze1x1', 'fire8/expand1x1', 'fire8/expand3x3',
        'fire9/squeeze1x1', 'fire9/expand1x1', 'fire9/expand3x3',
        'conv10',
    ]

    snet = SqueezeNet(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights)

    x = snet.outputs[0]
    x = GlobalAveragePooling2D()(x)
    # Use relu activation in the end to have all-zero probas
    out = Activation('relu', name='loss')(x)
    model = Model(inputs=snet.inputs, outputs=out)

    # Set some layers trainable
    for layer in model.layers:
        if layer.name in names_to_train:
            layer.trainable = True
        else:
            layer.trainable = False

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


import os
import sys

current_path = os.path.abspath(os.path.dirname(__file__))
squeezenet_path = os.path.abspath(os.path.join(current_path, 'KerasSqueezeNet', 'keras_squeezenet'))
if squeezenet_path not in sys.path:
    sys.path.append(squeezenet_path)

from keras.applications import ResNet50
from keras.optimizers import Adadelta, Adam, SGD
from keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from .keras_metrics import precision, recall


def get_resnet(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    weights = 'imagenet' if 'weights' not in params else params['weights']

    resnet = ResNet50(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights)

    # Not-trained layers :
    not_trained_layer_names = [
        'conv1', 'bn_conv1',
        'res2a_branch2a', 'bn2a_branch2a', 'res2a_branch2b', 'bn2a_branch2b',
        'res2a_branch2c', 'res2a_branch1', 'bn2a_branch2c', 'bn2a_branch1',
        'res2b_branch2a', 'bn2b_branch2a', 'res2b_branch2b', 'bn2b_branch2b',
        'res2b_branch2c', 'bn2b_branch2c', 'res2c_branch2a', 'bn2c_branch2a',
        'res2c_branch2b', 'bn2c_branch2b', 'res2c_branch2c', 'bn2c_branch2c',

        'res3a_branch2a', 'bn3a_branch2a', 'res3a_branch2b', 'bn3a_branch2b',
        'res3a_branch2c', 'res3a_branch1', 'bn3a_branch2c', 'bn3a_branch1',
        'res3b_branch2a', 'bn3b_branch2a', 'res3b_branch2b', 'bn3b_branch2b',
        'res3b_branch2c', 'bn3b_branch2c', 'res3c_branch2a', 'bn3c_branch2a',
        'res3c_branch2b', 'bn3c_branch2b', 'res3c_branch2c', 'bn3c_branch2c',
        'res3d_branch2a', 'bn3d_branch2a', 'res3d_branch2b', 'bn3d_branch2b',
        'res3d_branch2c', 'bn3d_branch2c',

        'res4a_branch2a', 'bn4a_branch2a', 'res4a_branch2b', 'bn4a_branch2b',
        'res4a_branch2c', 'res4a_branch1', 'bn4a_branch2c', 'bn4a_branch1',
        'res4b_branch2a', 'bn4b_branch2a', 'res4b_branch2b', 'bn4b_branch2b',
        'res4b_branch2c', 'bn4b_branch2c', 'res4c_branch2a', 'bn4c_branch2a',
        'res4c_branch2b', 'bn4c_branch2b', 'res4c_branch2c', 'bn4c_branch2c',
        'res4d_branch2a', 'bn4d_branch2a', 'res4d_branch2b', 'bn4d_branch2b',
        'res4d_branch2c', 'bn4d_branch2c', 'res4e_branch2a', 'bn4e_branch2a',
        'res4e_branch2b', 'bn4e_branch2b', 'res4e_branch2c', 'bn4e_branch2c',
        'res4f_branch2a', 'bn4f_branch2a', 'res4f_branch2b',
        'bn4f_branch2b', 'res4f_branch2c', 'bn4f_branch2c',

        'res5a_branch2a', 'bn5a_branch2a', 'res5a_branch2b', 'bn5a_branch2b',
        'res5a_branch2c', 'res5a_branch1', 'bn5a_branch2c', 'bn5a_branch1',
        'res5b_branch2a', 'bn5b_branch2a', 'res5b_branch2b', 'bn5b_branch2b',
        'res5b_branch2c', 'bn5b_branch2c',
        # 'res5c_branch2a', 'bn5c_branch2a', 'res5c_branch2b', 'bn5c_branch2b', 'res5c_branch2c', 'bn5c_branch2c',
    ]

    for layer in resnet.layers:
       if layer.name in not_trained_layer_names:
           layer.trainable = False

    x = resnet.outputs[0]
    x = Flatten()(x)
    out = Dense(n_classes, activation='sigmoid', name='tags')(x)

    model = Model(inputs=resnet.inputs, outputs=out)
    model.name = "ResNet50"

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


def get_resnet2(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    weights = 'imagenet' if 'weights' not in params else params['weights']

    resnet = ResNet50(input_shape=input_shape, classes=n_classes, include_top=False, weights=weights)

    # Not-trained layers :
    not_trained_layer_names = [
        'conv1', 'bn_conv1',
        'res2a_branch2a', 'bn2a_branch2a', 'res2a_branch2b', 'bn2a_branch2b',
        'res2a_branch2c', 'res2a_branch1', 'bn2a_branch2c', 'bn2a_branch1',
        'res2b_branch2a', 'bn2b_branch2a', 'res2b_branch2b', 'bn2b_branch2b',
        'res2b_branch2c', 'bn2b_branch2c', 'res2c_branch2a', 'bn2c_branch2a',
        'res2c_branch2b', 'bn2c_branch2b', 'res2c_branch2c', 'bn2c_branch2c',

        'res3a_branch2a', 'bn3a_branch2a', 'res3a_branch2b', 'bn3a_branch2b',
        'res3a_branch2c', 'res3a_branch1', 'bn3a_branch2c', 'bn3a_branch1',
        'res3b_branch2a', 'bn3b_branch2a', 'res3b_branch2b', 'bn3b_branch2b',
        'res3b_branch2c', 'bn3b_branch2c', 'res3c_branch2a', 'bn3c_branch2a',
        'res3c_branch2b', 'bn3c_branch2b', 'res3c_branch2c', 'bn3c_branch2c',
        'res3d_branch2a', 'bn3d_branch2a', 'res3d_branch2b', 'bn3d_branch2b',
        'res3d_branch2c', 'bn3d_branch2c',

        'res4a_branch2a', 'bn4a_branch2a', 'res4a_branch2b', 'bn4a_branch2b',
        'res4a_branch2c', 'res4a_branch1', 'bn4a_branch2c', 'bn4a_branch1',
        'res4b_branch2a', 'bn4b_branch2a', 'res4b_branch2b', 'bn4b_branch2b',
        'res4b_branch2c', 'bn4b_branch2c', 'res4c_branch2a', 'bn4c_branch2a',
        'res4c_branch2b', 'bn4c_branch2b', 'res4c_branch2c', 'bn4c_branch2c',
        'res4d_branch2a', 'bn4d_branch2a', 'res4d_branch2b', 'bn4d_branch2b',
        'res4d_branch2c', 'bn4d_branch2c', 'res4e_branch2a', 'bn4e_branch2a',
        'res4e_branch2b', 'bn4e_branch2b', 'res4e_branch2c', 'bn4e_branch2c',
        'res4f_branch2a', 'bn4f_branch2a', 'res4f_branch2b',
        'bn4f_branch2b', 'res4f_branch2c', 'bn4f_branch2c',

        'res5a_branch2a', 'bn5a_branch2a', 'res5a_branch2b', 'bn5a_branch2b',
        'res5a_branch2c', 'res5a_branch1', 'bn5a_branch2c', 'bn5a_branch1',
        'res5b_branch2a', 'bn5b_branch2a', 'res5b_branch2b', 'bn5b_branch2b',
        'res5b_branch2c', 'bn5b_branch2c',
        'res5c_branch2a', 'bn5c_branch2a', 'res5c_branch2b', 'bn5c_branch2b', 'res5c_branch2c', 'bn5c_branch2c',
    ]

    for layer in resnet.layers:
       if layer.name in not_trained_layer_names:
           layer.trainable = False

    x = resnet.outputs[0]
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(n_classes, activation='sigmoid', name='tags')(x)

    model = Model(inputs=resnet.inputs, outputs=out)
    model.name = "ResNet50_2"

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

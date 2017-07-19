
from keras.optimizers import Adadelta, Adam, SGD
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.models import Model
from .keras_metrics import precision, recall, f2


def get_mini_vgg_bn(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    final_activation = 'sigmoid'

    def _conv2d(input_layer, n_filters, padding='same', s_id=''):
        x = Conv2D(n_filters, kernel_size=(2, 2), padding=padding, name='conv_%s' % s_id)(input_layer)
        x = BatchNormalization(name='bn_%s' % s_id)(x)
        return Activation('relu')(x)

    def _block(input_layer, n_filters, s_id):
        x = _conv2d(input_layer, n_filters=n_filters, s_id='%s_1' % s_id)
        x = _conv2d(x, n_filters=n_filters, padding='valid', s_id='%s_2' % s_id)
        x = MaxPooling2D(pool_size=(2, 2), name='pool_%s' % s_id)(x)
        return Dropout(0.25)(x)

    inputs = Input(input_shape)
    x = inputs
    x = BatchNormalization()(x)

    x = _block(x, n_filters=32, s_id='b1')
    x = _block(x, n_filters=64, s_id='b2')
    x = _block(x, n_filters=128, s_id='b3')
    x = _block(x, n_filters=256, s_id='b4')

    x = Flatten()(x)
    x = Dense(512, activation='relu', name='d1')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation=final_activation, name='tags')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.name = "mini-VGG-BN"

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss=loss, optimizer=opt, metrics=[precision, recall, f2])
    return model


def get_mini_vgg_bn_2(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    final_activation = 'sigmoid'

    def _conv2d(input_layer, n_filters, padding='same', s_id=''):
        x = Conv2D(n_filters, kernel_size=(2, 2), padding=padding, name='conv_%s' % s_id)(input_layer)
        x = BatchNormalization(name='bn_%s' % s_id)(x)
        return Activation('relu')(x)

    def _block(input_layer, n_filters, s_id):
        x = _conv2d(input_layer, n_filters=n_filters, s_id='%s_1' % s_id)
        x = _conv2d(x, n_filters=n_filters, padding='valid', s_id='%s_2' % s_id)
        x = MaxPooling2D(pool_size=(2, 2), name='pool_%s' % s_id)(x)
        return Dropout(0.25)(x)

    inputs = Input(input_shape)
    x = inputs

    x = _block(x, n_filters=64, s_id='b1')
    x = _block(x, n_filters=128, s_id='b2')
    x = _block(x, n_filters=256, s_id='b3')
    x = _block(x, n_filters=512, s_id='b4')

    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='d1')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', name='d2')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation=final_activation, name='tags')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.name = "mini-VGG-BN-2"

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss=loss, optimizer=opt, metrics=[precision, recall, f2])
    return model


def get_mini_vgg(input_shape, n_classes, **params):
    """
    """
    optimizer = '' if 'optimizer' not in params else params['optimizer']
    lr = 0.01 if 'lr' not in params else params['lr']
    loss = '' if 'loss' not in params else params['loss']
    final_activation = 'sigmoid'

    def _conv2d(input_layer, n_filters, padding='same', s_id=''):
        x = Conv2D(n_filters, kernel_size=(2, 2), padding=padding, name='conv_%s' % s_id)(input_layer)
        return Activation('relu')(x)

    def _block(input_layer, n_filters, s_id):
        x = _conv2d(input_layer, n_filters=n_filters, s_id='%s_1' % s_id)
        x = _conv2d(x, n_filters=n_filters, padding='valid', s_id='%s_2' % s_id)
        x = MaxPooling2D(pool_size=(2, 2), name='pool_%s' % s_id)(x)
        return Dropout(0.25)(x)

    inputs = Input(input_shape)
    x = inputs
    x = BatchNormalization()(x)

    x = _block(x, n_filters=32, s_id='b1')
    x = _block(x, n_filters=64, s_id='b2')
    x = _block(x, n_filters=128, s_id='b3')
    x = _block(x, n_filters=256, s_id='b4')

    x = Flatten()(x)
    x = Dense(512, activation='relu', name='d1')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(n_classes, activation=final_activation, name='tags')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.name = "mini-VGG"

    if optimizer == 'adadelta':
        opt = Adadelta(lr=lr)
    elif optimizer == 'adam':
        opt = Adam(lr=lr)
    elif optimizer == 'sgd':
        opt = SGD(lr=lr, momentum=0.9, decay=0.00001, nesterov=True)
    else:
        opt = None

    if opt is not None:
        model.compile(loss=loss, optimizer=opt, metrics=[precision, recall, f2])
    return model

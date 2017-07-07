from math import floor

import warnings

from keras.models import Model
from keras.layers import Input, Concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.initializers import VarianceScaling

# https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua#L143
he_normal_fan_out = VarianceScaling(scale=2., mode='fan_in', distribution='normal')


WEIGHTS_PATH = ''
WEIGHTS_PATH_NO_TOP = ''


def DenseNet(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             classes=1000, **params):
    """
        Instantiate the DenseNet architecture for ImageNet

        DenseNet-BC : use_bottleneck=True, reduction=0.5
        DenseNet : use_bottleneck=False, reduction=1.0

        Code adapted from https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua

        # Arguments
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
            weights: one of `None` (random initialization)
                or "imagenet" (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or `(3, 224, 224)` (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 48.
                E.g. `(200, 200, 3)` would be one valid value.
            classes: optional number of classes to classify images
                into, only to be specified if `include_top` is True, and
                if no `weights` argument is specified.
        # Keyword Arguments
            depth: possible values {121, 161, 169, 201, 'custom'}, corresponds to DenseNet-121, DenseNet-169 and DenseNet-161
            stages: optional, used with type = 'custom', should have 4 integer values, e.g. (6, 12, 24, 32)
            growth_rate: number of filters to add per dense block, recommended values for ImageNet, k=32 or k=48
            reduction: reduction factor of transition blocks.
            dropout_rate: dropout rate
            use_bottleneck: use bottleneck layers
        # Returns
            A Keras model instance.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    depth_stages = {
        121: (6, 12, 24, 16),
        169: (6, 12, 32, 32),
        201: (6, 12, 48, 32),
        161: (6, 12, 36, 24),
    }
    depth = 121 if 'depth' not in params else params['depth']
    assert depth in depth_stages or depth == 'custom', "Unknown depth value. See doc for available type values"

    if weights == 'imagenet' and depth == 'custom':
        raise ValueError('Should not specify `weights` as imagenet with `depth=\'custom\'`')

    depth_growth_rate = {
        121: 32,
        169: 32,
        201: 32,
        161: 48,
    }
    if 'growth_rate' not in params:
        assert depth in depth_growth_rate, "Parameter depth should be: 121 or 169, 201, 161"
        growth_rate = depth_growth_rate[depth]
    else:
        growth_rate = params['growth_rate']

    stages = () if 'stages' not in params else params['stages']
    if depth == 'custom':
        assert len(stages) == 4, "Parameter stages should have 4 positive values"
        depth_stages[depth] = stages

    _params = dict(params)
    _params["growth_rate"] = growth_rate
    _params["reduction"] = 0.5 if 'reduction' not in _params else _params['reduction']
    _params["dropout_rate"] = 0.2 if 'dropout_rate' not in _params else _params['dropout_rate']
    _params["use_bottleneck"] = True if 'use_bottleneck' not in _params else _params['use_bottleneck']

    n_channels = 2 * growth_rate

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if K.image_data_format() == 'channels_last':
        _params["bn_axis"] = -1
    else:
        _params["bn_axis"] = 1

    _params["concat_axis"] = _params["bn_axis"]

    # Memory option
    _params["mem_option"] = 0

    # Initial transforms follow ResNet 224x224
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(n_channels, 7, 7, subsample=(2, 2), name='conv1',
                      use_bias=False, kernel_initializer=he_normal_fan_out)(x)
    x = BatchNormalization(axis=_params["bn_axis"], name='bn_conv1')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Dense-Block 1 and transition 56 x 56
    layer_id = "1"
    x = add_dense_block(x, stage=depth_stages[depth][0], layer_id=layer_id, **_params)
    n_outchannels = int(floor(x._keras_shape[_params['concat_axis']] * _params["reduction"]))
    x = add_transition(x, n_outchannels, layer_id=layer_id, **_params)

    # Dense-Block 2 and transition 28 x 28
    layer_id = "2"
    x = add_dense_block(x, stage=depth_stages[depth][1], layer_id=layer_id, **_params)
    n_outchannels = int(floor(x._keras_shape[_params['concat_axis']] * _params["reduction"]))
    x = add_transition(x, n_outchannels, layer_id=layer_id, **_params)

    # Dense - Block 3 and transition 14 x 14
    layer_id = "3"
    x = add_dense_block(x, stage=depth_stages[depth][2], layer_id=layer_id, **_params)
    n_outchannels = int(floor(x._keras_shape[_params['concat_axis']] * _params["reduction"]))
    x = add_transition(x, n_outchannels, layer_id=layer_id, **_params)

    # Dense - Block 4 and transition 7 x 7
    layer_id = "4"
    x = add_dense_block(x, stage=depth_stages[depth][3], layer_id=layer_id, **_params)

    if _params['mem_option'] >= 2:
        raise Exception("Not yet implemented")

    x = BatchNormalization(axis=_params['bn_axis'], name='transition_%s_bn1' % layer_id)(x)
    x = Activation('relu', name='transition_%s_relu1' % layer_id)(x)

    if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name="final_avg_pooling")(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='densenet{}'.format(depth))

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('densenet%s_weights_tf_dim_ordering_tf_kernels.h5' % depth,
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('densenet%s_weights_tf_dim_ordering_tf_kernels_notop.h5' % depth,
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)
        if K.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if K.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


def add_dense_block(input_layer, stage, layer_id, **params):
    x = input_layer
    for i in range(stage):
        x1 = add_layer(x, layer_id=layer_id + '_%i' % i, **params)
        x = Concatenate(axis=params['concat_axis'])([x, x1])
    return x


def add_layer(input_layer, **params):

    if params['mem_option'] >= 2:
        raise Exception("Not yet implemented")
        # x = dense_connect_layer_custom(input_layer, **params)
    else:
        x = dense_connect_layer_standard(input_layer, **params)
    return x


def add_transition(input_layer, n_outchannels, layer_id="",
                   bn_axis=-1, dropout_rate=0.0, **params):

    if params['mem_option'] >= 2:
        raise Exception("Not yet implemented")

    x = BatchNormalization(axis=bn_axis, name='transition_%s_bn1' % layer_id)(input_layer)
    x = Activation('relu', name='transition_%s_relu1' % layer_id)(x)
    x = Convolution2D(n_outchannels, 1, 1,
                      name="transition_%s_conv" % layer_id,
                      use_bias=False, kernel_initializer=he_normal_fan_out)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='transition_%s_dropout1' % layer_id)(x)
    x = AveragePooling2D(pool_size=(2, 2), name='transition_%s_pool' % layer_id)(x)
    return x


def dense_connect_layer_standard(input_layer, layer_id="",
                                 bn_axis=-1,
                                 use_bottleneck=True,
                                 growth_rate=32, dropout_rate=0.0, **params):

    x = BatchNormalization(axis=bn_axis, name='dcl_stand_%s_bn1' % layer_id)(input_layer)
    x = Activation('relu', name='dcl_stand_%s_relu1' % layer_id)(x)
    if use_bottleneck:
        x = Convolution2D(4 * growth_rate, 1, 1,
                          name="dcl_stand_%s_bottleneck_conv" % layer_id,
                          use_bias=False, kernel_initializer=he_normal_fan_out)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='dcl_stand_%s_dropout1' % layer_id)(x)
        x = BatchNormalization(axis=bn_axis, name='dcl_stand_%s_bn2' % layer_id)(x)
        x = Activation('relu', name='dcl_stand_%s_relu2' % layer_id)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(growth_rate, 3, 3,
                      name="dcl_stand_%s_conv1" % layer_id,
                      use_bias=False, kernel_initializer=he_normal_fan_out)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='dcl_stand_%s_dropout2' % layer_id)(x)
    return x

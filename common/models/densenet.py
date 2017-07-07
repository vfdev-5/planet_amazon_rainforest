from math import floor

from keras.models import Model
from keras.layers import Input, Concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.applications.imagenet_utils import _obtain_input_shape


def DenseNet(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             classes=1000, **params
             #nb_dense_block=4, growth_rate=32,
             #reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
             #classes=1000, weights='imagenet', include_top=True
             ):
    '''
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

        # Keyword Arguments
            depth: possible values {121, 161, 169, 201, 'custom'}, corresponds to DenseNet-121, DenseNet-169 and DenseNet-161
            stages: optional, used with type = 'custom', should have 4 integer values, e.g. (6, 12, 24, 32)

            growth_rate: number of filters to add per dense block, recommended values for ImageNet, k=32 or k=48
            reduction: reduction factor of transition blocks.

            nb_dense_block: number of dense blocks to add to end
            nb_filter: initial number of filters

            dropout_rate: dropout rate
            weight_decay: weight decay factor
            classes: optional number of classes to classify images
            weights_path: path to pre-trained weights
        # Returns
            A Keras model instance.
    '''

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

    reduction = 0.5 if 'reduction' not in params else params['reduction']
    dropout_rate = 0.2 if 'dropout_rate' not in params else params['dropout_rate']
    use_bottleneck = True if 'use_bottleneck' not in params else params['use_bottleneck']

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
        bn_axis = 3
    else:
        bn_axis = 1

    # Memory option
    mem_option = 0

    # Initial transforms follow ResNet 224x224
    x = ZeroPadding2D((3, 3))(img_input)
    x = Convolution2D(n_channels, 7, 7, subsample=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Dense-Block 1 and transition 56 x 56
    x, n_channels = add_dense_block(x, n_channels,
                                    stage=depth_stages[depth][0],
                                    bn_axis=bn_axis,
                                    concat_axis=bn_axis,
                                    mem_option=mem_option,
                                    growth_rate=growth_rate)
    x = add_transition(x, n_channels, int(floor(n_channels * reduction)))
    n_channels = int(floor(n_channels * reduction))

    # Dense-Block 2 and transition 28 x 28
    x, n_channels = add_dense_block(x, n_channels, stage=depth_stages[depth][1])
    x = add_transition(x, n_channels, int(floor(n_channels * reduction)))
    n_channels = int(floor(n_channels * reduction))






    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv'+str(final_stage)+'_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv'+str(final_stage)+'_blk_scale')(x)
    x = Activation('relu', name='relu'+str(final_stage)+'_blk')(x)
    x = GlobalAveragePooling2D(name='pool'+str(final_stage))(x)

    x = Dense(classes, name='fc6')(x)
    x = Activation('softmax', name='prob')(x)

    model = Model(img_input, x, name='densenet')

    if weights_path is not None:
        model.load_weights(weights_path)

    return model


def add_dense_block(input_layer, n_channels, stage, concat_axis, growth_rate, **params):
    x = input_layer
    for i in range(stage):
        x1 = add_layer(x, n_channels=n_channels, **params)
        x = Concatenate(axis=concat_axis)([x, x1])
        n_channels += growth_rate
    return x, n_channels


def add_layer(input_layer, n_channels, bn_axis=-1, mem_option=0, **params):
    if mem_option >= 3:
        raise Exception("Not yet implemented")
        # x = DenseConnectLayerCustom(n_channels)(x)
    else:
        x = dense_connect_layer_standard(input_layer, bn_axis, **params)
    return x


def add_transition(input_layer, n_channels, **params):
    pass


def dense_connect_layer_standard(input_layer, layer_id="",
                                 bn_axis=-1,
                                 use_bottleneck=True,
                                 growth_rate=32, dropout_rate = 0.0):

    x = BatchNormalization(axis=bn_axis, name='dcl_stand_%s_bn1' % layer_id)(input_layer)
    x = Activation('relu', name='dcl_stand_%s_relu1' % layer_id)(x)
    if use_bottleneck:
        x = Convolution2D(4 * growth_rate, 1, 1,
                          name="dcl_stand_%s_bottleneck_conv" % layer_id)(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name='dcl_stand_%s_dropout1' % layer_id)(x)
        x = BatchNormalization(axis=bn_axis, name='dcl_stand_%s_bn2' % layer_id)(x)
        x = Activation('relu', name='dcl_stand_%s_relu2')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(growth_rate, 3, 3,
                      name="dcl_stand_%s_conv1" % layer_id)(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name='dcl_stand_%s_dropout2' % layer_id)(x)
    return x




def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x1_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution2D(inter_channel, 1, 1, name=conv_name_base+'_x1', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_x2_scale')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution2D(nb_filter, 3, 3, name=conv_name_base+'_x2', bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base+'_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base+'_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, name=conv_name_base, bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = merge([concat_feat, x], mode='concat', concat_axis=concat_axis,
                            name='concat_'+str(stage)+'_'+str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


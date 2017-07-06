
import os
import sys
from datetime import datetime

import numpy as np

from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, Callback
import keras.backend as K
from keras import __version__ as keras_version


# Local repos:
local_repos_path = os.path.abspath(os.path.dirname(__file__))

keras_contrib_path = os.path.join(local_repos_path, "KerasContrib", "keras_contrib")
if keras_contrib_path not in sys.path:
    sys.path.append(keras_contrib_path)

imgaug_contrib_path = os.path.join(local_repos_path, "imgaug", "imgaug")
if imgaug_contrib_path not in sys.path:
    sys.path.append(imgaug_contrib_path)


from preprocessing.image.generators import ImageDataGenerator
from imgaug.imgaug import augmenters as iaa

from data_utils import GENERATED_DATA, unique_tags
from xy_providers import image_class_labels_provider
from metrics import score
from postproc import pred_threshold


def random_imgaug(x, seq):
    return seq.augment_images([x, ])[0]


def get_train_imgaug_seq(seed):
    train_seq = iaa.Sequential([
        iaa.Sometimes(0.45, iaa.Sharpen(alpha=0.9, lightness=(0.5, 1.15))),
        iaa.Sometimes(0.45, iaa.ContrastNormalization(alpha=(0.75, 1.15))),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255), per_channel=True)),
        iaa.Affine(translate_px=(-25, 25),
                   scale=(0.85, 1.15),
                   rotate=(-65, 65),
                   mode='reflect'),
        # iaa.Add(value=(-35, 35), per_channel=True),  # Probably, can change nature of label
    ],
        random_order=True,
        random_state=seed
    )
    return train_seq


def get_id_imgaug_seq():
    return iaa.Sequential()


def get_val_imgaug_seq(seed):
    val_seq = iaa.Sequential([
        iaa.Affine(translate_px=(-25, 25),
                   scale=(0.85, 1.15),
                   rotate=(-45, 45),
                   mode='reflect'),
    ],
        random_order=True,
        random_state=seed
    )
    return val_seq


def get_gen_flow(id_type_list, **params):

    seed = params.get('seed')
    normalize_data = params.get('normalize_data')
    normalization = params.get('normalization')
    xy_provider = params.get('xy_provider')
    save_prefix = params.get('save_prefix')
    imgaug_seq = params.get('imgaug_seq')
    batch_size = params.get('batch_size')
    verbose = params.get('verbose')

    assert seed is not None, "seed is needed"
    assert normalize_data is not None, "normalize_data is needed"
    assert normalization is not None, "normalization is needed"
    assert batch_size is not None, "batch_size is needed"
    assert 'image_size' in params, "image_size is needed"

    if normalize_data and (normalization == '' or normalization == 'from_save_prefix'):
        assert save_prefix is not None, "save_prefix is needed"

    if verbose is None:
        verbose = 0

    if xy_provider is None:
        xy_provider = image_class_labels_provider

    if hasattr(K, 'image_data_format'):
        channels_first = K.image_data_format() == 'channels_first'
    elif hasattr(K, 'image_dim_ordering'):
        channels_first = K.image_dim_ordering() == 'th'
    else:
        raise Exception("Failed to find backend data format")

    def _random_imgaug(x):
        return random_imgaug(255.0 * x, imgaug_seq) * 1.0/255.0

    pipeline = ('random_transform',)
    if imgaug_seq is not None:
        pipeline += (_random_imgaug, )
    pipeline += ('standardize', )

    gen = ImageDataGenerator(pipeline=pipeline,
                             featurewise_center=normalize_data,
                             featurewise_std_normalization=normalize_data,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode='reflect')

    if normalize_data:
        if normalization == '':
            print("\n-- Fit stats of train dataset")
            gen.fit(xy_provider(id_type_list,
                                test_mode=True,
                                channels_first=channels_first, **params),
                    len(id_type_list),
                    augment=True,
                    seed=seed,
                    save_to_dir=GENERATED_DATA,
                    save_prefix=save_prefix,
                    batch_size=4,
                    verbose=verbose)
        elif normalization == 'inception' or normalization == 'xception':
            # Preprocessing of Xception: keras/applications/xception.py
            if verbose > 0:
                print("Image normalization: ", normalization)
            gen.mean = 0.5
            gen.std = 0.5
        elif normalization == 'resnet' or normalization == 'vgg':
            if verbose > 0:
                print("Image normalization: ", normalization)
            gen.std = 1.0 / 255.0  # Rescale to [0.0, 255.0]
            m = np.array([123.68, 116.779, 103.939]) / 255.0  # RGB
            if channels_first:
                m = m[:, None, None]
            else:
                m = m[None, None, :]
            gen.mean = m
        elif normalization == 'from_save_prefix':
            assert len(save_prefix) > 0, "WTF"
            # Load mean, std, principal_components if file exists
            filename = os.path.join(GENERATED_DATA, save_prefix + "_stats.npz")
            assert os.path.exists(filename), "WTF"
            if verbose > 0:
                print("Load existing file: %s" % filename)
            npzfile = np.load(filename)
            gen.mean = npzfile['mean']
            gen.std = npzfile['std']

    flow = gen.flow(xy_provider(id_type_list,
                                channels_first=channels_first,
                                **params),
                    # Ensure that all batches have the same size
                    (len(id_type_list) // batch_size) * batch_size,
                    seed=seed,
                    batch_size=batch_size)
    return gen, flow


def exp_decay(epoch, lr=1e-3, a=0.925, init_epoch=0):
    return lr * np.exp(-(1.0 - a) * (epoch + init_epoch))


def step_decay(epoch, lr=1e-3, base=2.0, period=50, init_epoch=0):
    return lr * base ** (-np.floor((epoch + init_epoch) * 1.0 / period))


def write_info(filename, **kwargs):
    with open(filename, 'w') as f:
        for k in kwargs:
            f.write("{}: {}\n".format(k, kwargs[k]))


class EpochValidationCallback(Callback):

    def __init__(self, val_id_type_list, **params):
        super(EpochValidationCallback, self).__init__()
        self.val_id_type_list = val_id_type_list
        self.ev_params = params
        self.scores = []
        assert 'seed' in self.ev_params, "Need seed, params: {}".format(self.ev_params)

    def on_epoch_begin(self, epoch, logs=None):
        score = classification_validate(self.model, self.val_id_type_list, verbose=0, **self.ev_params)
        print("\nEpoch validation: score = %f \n" % score)
        self.scores.append(score)


def classification_train(model,
                         train_id_type_list,
                         val_id_type_list,
                         **params):

    assert 'batch_size' in params, "Need batch_size"
    assert 'save_prefix' in params, "Need save_prefix"
    assert 'nb_epochs' in params, "Need nb_epochs"
    assert 'seed' in params, "Need seed"
    assert 'normalize_data' in params, "Need normalize_data"

    samples_per_epoch = len(train_id_type_list) if 'samples_per_epoch' not in params else params['samples_per_epoch']
    nb_val_samples = len(val_id_type_list) if 'nb_val_samples' not in params else params['nb_val_samples']
    lr_decay_f = None if 'lr_decay_f' not in params else params['lr_decay_f']

    save_prefix = params['save_prefix']
    batch_size = params['batch_size']
    nb_epochs = params['nb_epochs']
    seed = params['seed']
    normalize_data = params['normalize_data']
    if normalize_data:
        assert 'normalization' in params, "Need normalization"
        normalization = params['normalization']
    else:
        normalization = None

    samples_per_epoch = (samples_per_epoch // batch_size + 1) * batch_size
    nb_val_samples = (nb_val_samples // batch_size + 1) * batch_size

    output_path = params['output_path'] if 'output_path' in params else GENERATED_DATA
    weights_path = os.path.join(output_path, "weights")
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    weights_filename = os.path.join(weights_path,
                                    save_prefix + "_{epoch:02d}_val_loss={val_loss:.4f}")

    metrics_names = list(model.metrics_names)
    metrics_names.remove('loss')
    for mname in metrics_names:
        weights_filename += "_val_%s={val_%s:.4f}" % (mname, mname)
    weights_filename += ".h5"

    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_loss',
                                       save_best_only=True, save_weights_only=True)
    now = datetime.now()
    info_filename = os.path.join(weights_path,
                                 'training_%s_%s.info' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M"))))

    write_info(info_filename, **params)

    csv_logger = CSVLogger(os.path.join(weights_path,
                                        'training_%s_%s.log' % (save_prefix, str(now.strftime("%Y-%m-%d-%H-%M")))))

    epoch_validation = EpochValidationCallback(val_id_type_list, **params)

    callbacks = [model_checkpoint, csv_logger, epoch_validation]
    if lr_decay_f is not None:
        assert 'lr_kwargs' in params, "Need lr_kwargs"
        _lr_decay_f = lambda e: lr_decay_f(epoch=e, **params['lr_kwargs'])
        lrate = LearningRateScheduler(_lr_decay_f)
        callbacks.append(lrate)

    print("\n-- Training parameters: %i, %i, %i, %i" % (batch_size, nb_epochs, samples_per_epoch, nb_val_samples))
    print("\n-- Fit model")

    class_weight = params.get('class_weight')
    verbose = 1 if 'verbose' not in params else params['verbose']

    try:

        train_seq = get_train_imgaug_seq(seed)
        train_gen, train_flow = get_gen_flow(id_type_list=train_id_type_list,
                                             imgaug_seq=train_seq,
                                             **params)
        if normalize_data and normalization == '':
            params['normalization'] = 'from_save_prefix'

        val_seq = get_val_imgaug_seq(seed)
        val_gen, val_flow = get_gen_flow(id_type_list=val_id_type_list,
                                         imgaug_seq=val_seq,
                                         **params)

        np.random.seed(seed)
        # New or old Keras API
        if int(keras_version[0]) == 2:
            print("- New Keras API found -")
            history = model.fit_generator(generator=train_flow,
                                          steps_per_epoch=(samples_per_epoch // batch_size),
                                          epochs=nb_epochs,
                                          validation_data=val_flow,
                                          validation_steps=(nb_val_samples // batch_size),
                                          callbacks=callbacks,
                                          class_weight=class_weight,
                                          verbose=verbose)
        else:
            history = model.fit_generator(generator=train_flow,
                                          samples_per_epoch=samples_per_epoch,
                                          nb_epoch=nb_epochs,
                                          validation_data=val_flow,
                                          nb_val_samples=nb_val_samples,
                                          callbacks=callbacks,
                                          class_weight=class_weight,
                                          verbose=verbose)
        return history

    except KeyboardInterrupt:
        pass


def rmse(y):
    return np.sqrt(np.mean(np.power(y, 2.0)))

from hashlib import sha256

def hash(y):
    return sha256(y).hexdigest()

def classification_validate(model,
                            val_id_type_list,
                            **params):

    assert 'seed' in params, "Need seed, params = {}".format(params)
    verbose = 1 if 'verbose' not in params else params['verbose']

    # val_seq = get_val_imgaug_seq(params['seed'])
    val_seq = None
    val_gen, val_flow = get_gen_flow(id_type_list=val_id_type_list,
                                     imgaug_seq=val_seq,
                                     test_mode=True, **params)

    y_true_total = np.zeros((len(val_id_type_list), len(unique_tags)))
    y_pred_total = np.zeros_like(y_true_total)
    counter = 0
    for x, y_true, info in val_flow:
        s = y_true.shape[0]
        if verbose > 0:
            print("-- %i / %i" % (s *counter, len(val_id_type_list)))
            print("-- ", hash(y_true), y_true)

        start = counter * s
        end = min((counter + 1) * s, len(val_id_type_list))
        y_true_total[start:end, :] = y_true

        y_pred = model.predict(x)
        y_pred2 = pred_threshold(y_pred)
        y_pred_total[start:end, :] = y_pred2

        counter += 1

    print("rmse: ", rmse(y_true_total), rmse(y_pred_total), rmse(y_true_total - y_pred_total))
    print("Hashes: ", hash(y_true_total), hash(y_pred_total))
    total_score = score(y_true_total, y_pred_total)
    if verbose > 0:
        print("Total score : ", total_score)
    return total_score

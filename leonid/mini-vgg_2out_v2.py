# based on https://www.kaggle.com/arsenyinfo/end-to-end-pipeline-with-vgg-and-augmentation
from itertools import chain
from threading import Lock
import logging
from os import listdir, path, environ

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

file_name = 'mini-vgg_2out_v2'

try:
    if __file__: print(__file__)
except NameError:
    __file__ = 'leonid/{}.py'.format(file_name)

# Local repos:
import sys
local_repos_path = path.abspath(path.dirname(__file__))

common_path = path.abspath(path.join(path.dirname(__file__), '..', 'common'))
if common_path not in sys.path:
    sys.path.append(common_path)
    sys.path


import h5py
import numpy as np
import pandas as pd
import joblib
from skimage.io import imread
from skimage.transform import rescale, rotate

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Input, MaxPooling2D, Conv2D
from keras import backend as K
#from imgaug.imgaug import augmenters as iaa
from imgaug import augmenters as iaa
import random
import tensorflow as tf

random_seed = 2017
random.seed(random_seed)
np.random.seed(random_seed)


logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

MAIN_DIR = path.abspath(path.join(path.dirname(__file__), '..', 'input'))

USE_TIFF = False

TRAIN_LABELS = path.join(MAIN_DIR, 'train_v2.csv')
TRAIN_DIR = path.join('train', 'tif') if USE_TIFF else path.join('train', 'jpg')
TEST_DIR = path.join('test', 'tif') if USE_TIFF else path.join('test', 'jpg')
# please copy all test images in one directory

TRAIN_DIR = path.join(MAIN_DIR, TRAIN_DIR, '')
TEST_DIR = path.join(MAIN_DIR, TEST_DIR, '')

DTYPE = np.float16

SOURCE_SIZE = 256
CROP_SIZE = SOURCE_SIZE
NEW_SIZE = 64
#SCALE_COEFF = NEW_SIZE / 256
SCALE_COEFF = NEW_SIZE / 256
N_FOLDS = 3

def f2_score(y_true, y_pred):
    # from https://www.kaggle.com/teasherm/keras-metric-for-f-score-tf-only
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)
# Ex. usage
# model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=[f2_score])

def stratified_kfold_sampling(Y, n_splits=5, num_classes=17, random_state=random_seed):
    train_folds = [[] for _ in range(n_splits)]
    valid_folds = [[] for _ in range(n_splits)]
    inx = np.arange(len(Y))

    for cl in range(0, num_classes):
        sss = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state+cl)

        for fold, (train_index, test_index) in enumerate(sss.split(inx, Y[:,cl])):
            b_train_inx, b_valid_inx = inx[train_index], inx[test_index]

            # to ensure there is no repetetion within each split and between the splits
            train_folds[fold] = train_folds[fold] + list(set(list(b_train_inx)) - \
                               set(train_folds[fold]) - set(valid_folds[fold]))
            valid_folds[fold] = valid_folds[fold] + list(set(list(b_valid_inx)) - \
                               set(train_folds[fold]) - set(valid_folds[fold]))

    return np.array(train_folds), np.array(valid_folds)


def form_double_batch(X, y1, y2, batch_size):
    idx = np.random.randint(0, X.shape[0], int(batch_size))
    return X[idx], y1[idx], y2[idx]


def rotate_determined(img):
    img1 = img
    img2 = rotate(img, 90, preserve_range=True)
    img3 = rotate(img, 180, preserve_range=True)
    img4 = rotate(img, 270, preserve_range=True)
    arr = np.array([img1, img2, img3, img4]).astype(np.float16)
    return arr


def get_rotate_angle():
    return np.random.choice([0, 90, 180, 270])


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


@threadsafe_generator
def double_batch_generator(X, y1, y2, batch_size):
    seq = iaa.Sequential([iaa.Sometimes(.8, iaa.Affine(rotate=get_rotate_angle(),
                                                       mode='reflect')),
                          iaa.Fliplr(p=.25)
                          ],
                         random_order=False)

    while True:
        x_batch, y1_batch, y2_batch = form_double_batch(X, y1, y2, batch_size)
        new_x_batch = seq.augment_images(x_batch)
        new_x_batch = np.array(new_x_batch).astype('float16')
        yield new_x_batch, {'labels': y1_batch, 'weather': y2_batch}


def process_image(fname):
    img = rescale(imread(fname), SCALE_COEFF, preserve_range=True, mode='reflect')
    return img.astype(DTYPE)


class Dataset:
    def __init__(self, batch_size=64, fold=2):
        self.df = pd.read_csv(TRAIN_LABELS)
        self.train_folds, self.test_folds = self.get_folds()
        self.fold = fold
        self.batch_size = batch_size
        self.labels, self.reverse_labels, self.weather, self.reverse_weather = self.get_labels()

    @staticmethod
    def get_folds():
        train_files = np.array(listdir(TRAIN_DIR))
##        kf = KFold(n_splits=n_folds)
##        for train_index, test_index in kf.split(_trainval_id_type_list):
#        train_folds, valid_folds = stratified_kfold_sampling(Y, n_splits=N_FOLDS)
#        for train_index, test_index in zip(train_folds, valid_folds):
##        print("TRAIN:", train_index, "TEST:", test_index)

        folder = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        trains, tests = zip(*folder.split(train_files))
        return trains, tests

    def get_labels(self):
        labels = self.df.tags.values
        weather = {'partly_cloudy', 'clear', 'cloudy', 'haze'}
        labels = list(set(chain(*[x.split(' ') for x in labels])) - weather)
        weather = list(weather)

        weather.sort()
        labels.sort()
        labels = {name: i for i, name in enumerate(labels)}
        reverse_labels = {i: name for i, name in enumerate(labels)}
        weather = {name: i for i, name in enumerate(weather)}
        reverse_weather = {i: name for i, name in enumerate(weather)}

        return labels, reverse_labels, weather, reverse_weather

    def cache_train(self):
        logger.info('Creating cache file for train')
        file = h5py.File('train.h5', 'w')
        train_files = listdir(TRAIN_DIR)

        x_data = file.create_dataset('train_x', shape=(len(train_files), NEW_SIZE, NEW_SIZE, 3), dtype=DTYPE)
        y_weather = file.create_dataset('train_weather', shape=(len(train_files), 4), dtype=DTYPE)
        y_labels = file.create_dataset('train_labels', shape=(len(train_files), 13), dtype=DTYPE)
        x_data_cropped = file.create_dataset('train_x_cropped', shape=(len(train_files) * 4, NEW_SIZE, NEW_SIZE, 3), dtype=DTYPE)
        names = file.create_dataset('train_names', shape=(len(train_files) * 4,), dtype=h5py.special_dtype(vlen=str))

        for i, (x, y_l, y_w, fname) in enumerate(self.get_images()):
            x_data[i, :, :, :] = x
            y_weather[i, :] = y_w
            y_labels[i, :] = y_l

            for j, img_cropped in enumerate(rotate_determined(x)):
                x_data_cropped[4 * i + j, :, :, :] = img_cropped
                names[4 * i + j] = fname

        file.close()

    def cache_test(self):
        logger.info('Creating cache file for test')
        file = h5py.File('test.h5', 'w')
        test_files = listdir(TEST_DIR)

        x_data = file.create_dataset('test_x', shape=(len(test_files) * 4, NEW_SIZE, NEW_SIZE, 3), dtype=DTYPE)
        x_names = file.create_dataset('test_names', shape=(len(test_files) * 4,), dtype=h5py.special_dtype(vlen=str))

        images = [(f, process_image(path.join(TEST_DIR, f))) for f in listdir(TEST_DIR)]

        for i, (f, img) in enumerate(images):
            for j, img_cropped in enumerate(rotate_determined(img)):
                x_data[4 * i + j, :, :, :] = img_cropped
                x_names[4 * i + j] = f
        file.close()

    def update_fold(self):
        if self.fold + 1 < len(self.train_folds):
            self.fold += 1
            logger.info('Switching to fold {}'.format(self.fold))
            return self.fold

        logger.info('It was a final fold')
        return

    def get_train(self, fold):
        try:
            file = h5py.File('train.h5', 'r')
        except OSError:
            self.cache_train()
            file = h5py.File('train.h5', 'r')

        x_data = file['train_x_cropped']
        x_names = file['train_names']
        idx = self.test_folds[fold]
        idx = np.hstack([np.array([4 * x, 4 * x + 1, 4 * x + 2, 4 * x + 3]) for x in idx]).tolist()
        return x_data[idx], x_names[idx]

    def get_test(self):
        try:
            file = h5py.File('test.h5', 'r')
        except OSError:
            self.cache_test()
            file = h5py.File('test.h5', 'r')

        x_data = file['test_x']
        x_names = file['test_names']

        return x_data, x_names

    def make_double_generator(self, use_train=True, batch_size=None):
        try:
            file = h5py.File('train.h5', 'r')
        except OSError:
            self.cache_train()
            file = h5py.File('train.h5', 'r')

        idx = self.train_folds[self.fold] if use_train else self.test_folds[self.fold]
        x_data = file['train_x']
        y_data_labels = file['train_labels']
        y_data_weather = file['train_weather']
        x_data, y_data_labels, y_data_weather = map(lambda x: x[idx.tolist()][:],
                                                    (x_data, y_data_labels, y_data_weather))

        return double_batch_generator(x_data, y_data_labels, y_data_weather,
                                      batch_size if batch_size else self.batch_size)

    def encode_target(self, tags):
        target_labels = np.zeros(len(self.labels))
        target_weather = np.zeros(len(self.weather))
        for tag in tags.split(' '):
            if tag in self.labels:
                target_labels[self.labels[tag]] = 1
            else:
                target_weather[self.weather[tag]] = 1
        return target_labels, target_weather

    def get_images(self):
        pd.read_csv(TRAIN_LABELS)
        images_dir = TRAIN_DIR

        ext = 'tif' if USE_TIFF else 'jpg'
        for i, row in self.df.iterrows():
            fname = '{}{}.{}'.format(images_dir, row.image_name, ext)
            x = process_image(fname)
            y_label, y_weather = map(lambda y: y.astype(np.int8), self.encode_target(row.tags))

            if not i % 1000:
                logger.info('{} images loaded'.format(i))

            yield x, y_label, y_weather, row.image_name


class Master:
    def __init__(self, batch_size=64, fold=0):
        self.batch_size = batch_size
        self.dataset = Dataset(batch_size=batch_size, fold=fold)
        self.fold = fold

    def get_callbacks(self, name):
        model_checkpoint = ModelCheckpoint('networks/{}_current_{}.h5'.format(name, self.fold),
                                           monitor='val_loss',
                                           save_best_only=True, verbose=0)
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        reducer = ReduceLROnPlateau(min_lr=1e-6, verbose=1, factor=0.1, patience=4)
        return [model_checkpoint, es, reducer]

    def get_vgg(self, shape):
#        vgg = VGG16(include_top=False, weights='imagenet', input_shape=shape)
#        vgg.layers = vgg.layers[:15]
#        gap = GlobalAveragePooling2D()(vgg.output)

        def _conv2d(input_layer, n_filters, padding='same', s_id=''):
            x = Conv2D(n_filters, kernel_size=(2, 2), padding=padding,
                       name='conv_%s' % s_id, activation='relu')(input_layer)
            return x

        def _block(input_layer, n_filters, s_id):
            x = _conv2d(input_layer, n_filters=n_filters, s_id='%s_1' % s_id)
            x = _conv2d(x, n_filters=n_filters, padding='valid', s_id='%s_2' % s_id)
            x = MaxPooling2D(pool_size=(2, 2), name='pool_%s' % s_id)(x)
            return Dropout(0.25)(x)

        inputs = Input(shape)
        x = inputs
        x = BatchNormalization()(x)

        x = _block(x, n_filters=32, s_id='b1')
        x = _block(x, n_filters=64, s_id='b2')
        x = _block(x, n_filters=128, s_id='b3')
        x = _block(x, n_filters=256, s_id='b4')

        gap = GlobalAveragePooling2D()(x)

        drop = Dropout(0.3)(gap)
        dense = Dense(1024)(drop)
        dense = LeakyReLU(alpha=.01)(dense)
        drop2 = Dropout(0.3)(dense)
        dense2 = Dense(128)(drop2)
        dense2 = LeakyReLU(alpha=.01)(dense2)
        out_labels = Dense(13, activation='sigmoid', name='labels')(dense2)
        out_weather = Dense(4, activation='softmax', name='weather')(dense2)

#        model = Model(inputs=vgg.input, outputs=[out_labels, out_weather])
        model = Model(inputs=inputs, outputs=[out_labels, out_weather])
        model.name = 'mini-vgg_2out_Adam'
        model.compile(optimizer=Adam(clipvalue=3),
                      loss={'labels': 'binary_crossentropy', 'weather': 'categorical_crossentropy'},
                      metrics=[f2_score]
                      )
        return model

    def fit(self):
        base_size = NEW_SIZE if NEW_SIZE else 256
        shape = (base_size, base_size, 4) if USE_TIFF else (base_size, base_size, 3)

        model = self.get_vgg(shape)

        logger.info('Fitting started')

        model.fit_generator(self.dataset.make_double_generator(use_train=True),
                            epochs=500,
                            steps_per_epoch=500,
                            validation_data=self.dataset.make_double_generator(use_train=True),
                            workers=4,
                            validation_steps=100,
                            callbacks=self.get_callbacks(model.name)
                            )

        new_fold = self.dataset.update_fold()
        if new_fold:
            self.fold = new_fold
            K.clear_session()
            self.fit()

    def _wrap_prediction(self, name, pred_l, pred_w):
        pred = {self.dataset.reverse_labels[i]: pred_l[i] for i in range(pred_l.shape[0])}
        pred.update({self.dataset.reverse_weather[i]: pred_w[i] for i in range(pred_w.shape[0])})
        pred['image_name'] = name.split('.')[0]
        return pred

    def wrap_predicitions(self, names, labels, weather):
        return joblib.Parallel(n_jobs=8, backend='threading')(
            joblib.delayed(self._wrap_prediction)(name, pred_l, pred_w)
            for name, pred_l, pred_w in zip(names, labels, weather))

    def make_predictions(self):
        test_data, test_names = self.dataset.get_test()

        test_result = []
        train_result = []

        for fold in range(N_FOLDS):
            logger.info('Prediction started for fold {}'.format(fold))
            model = load_model('networks/' + 'mini-vgg_2out_Adam' + '_current_{}.h5'.format(fold))

            labels_test, weather_test = model.predict(test_data, batch_size=96)

            train_data, train_names = self.dataset.get_train(fold)
            labels_train, weather_train = model.predict(train_data, batch_size=96, verbose=1)

            labels_test, weather_test, labels_train, weather_train = map(lambda x: x.astype('float16'),
                                                                         (labels_test, weather_test,
                                                                          labels_train, weather_train))

            logger.info('Data transformation started for fold {}'.format(fold))

            test_result += list(self.wrap_predicitions(test_names, labels_test, weather_test))
            train_result += list(self.wrap_predicitions(train_names, labels_train, weather_train))

            K.clear_session()

        train_result, test_result = map(lambda x: pd.DataFrame(x).groupby(['image_name']).agg(np.mean).reset_index(),
                                        (train_result, test_result))

        train_result.to_csv('train_probs.csv', index=False)
        test_result.to_csv('test_probs.csv', index=False)

        final = []
        threshold = .2
        for _, row in test_result.iterrows():
            row = row.to_dict()
            name = row.pop('image_name')
            tags = [k for k, v in row.items() if v > threshold]
            final.append({'image_name': name,
                          'tags': ' '.join(tags)})
        pd.DataFrame(final).to_csv('result.csv.gz', index=False, compression='gzip')


if __name__ == '__main__':
    master = Master(batch_size=64, fold=0)
    master.fit()
#    master.make_predictions()

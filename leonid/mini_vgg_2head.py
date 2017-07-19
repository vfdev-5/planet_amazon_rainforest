# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 10:17:04 2017

@author: Leonid Sinev
"""
__author__ = "LSinev: https://www.kaggle.com/lsinev"

name = 'mini_vgg_2head'

import os
import sys

try:
    if __file__: exit
except NameError:
    __file__ = 'leonid/{}.py'.format(name)

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input'))
FILE_FOLDER = os.path.abspath(os.path.dirname(__file__))

if not FILE_FOLDER in sys.path:
    sys.path.append(FILE_FOLDER)

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import AveragePooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import time

from tqdm import tqdm

import h5py # for saving later usable files

import gc
from glob import glob

from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import cv2
import bcolz
#from joblib import Parallel, delayed
import image_ml_ext #modified ImageDataGenerator

import random
random_seed = 2017
random.seed(random_seed)
np.random.seed(random_seed)

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


def get_mini_vgg(input_shape, n_classes, **kwargs):
    """
    """
    lr = 0.01 if 'lr' not in kwargs else kwargs['lr']
    lr_decay = 0.001 if 'lr_decay' not in kwargs else kwargs['lr_decay']
    loss = 'binary_crossentropy' if 'loss' not in kwargs else kwargs['loss']
    final_activation = 'sigmoid'

    def _conv2d(input_layer, n_filters, padding='same', s_id=''):
        x = Conv2D(n_filters, kernel_size=(2, 2), padding=padding,
                   name='conv_%s' % s_id, activation='relu')(input_layer)
        return x

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
    model.name = "mini-VGG_Adam"

    opt = Adam(lr=lr,
               decay=lr_decay)

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy', f2_score])
    return model

def get_mini_vgg_2head(input_shape, **kwargs):
    """
    """
    lr = 0.01 if 'lr' not in kwargs else kwargs['lr']
    lr_decay = 0.001 if 'lr_decay' not in kwargs else kwargs['lr_decay']
    loss = 'binary_crossentropy' if 'loss' not in kwargs else kwargs['loss']
    final_activation = 'sigmoid'

    def _conv2d(input_layer, n_filters, padding='same', s_id=''):
        x = Conv2D(n_filters, kernel_size=(2, 2), padding=padding,
                   name='conv_%s' % s_id, activation='relu')(input_layer)
        return x

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

#    x = Flatten()(x)
#    x = Dense(512, activation='relu', name='d1')(x)
#    x = Dropout(0.5)(x)
#    outputs = Dense(n_classes, activation=final_activation, name='tags')(x)

    x = BatchNormalization(name='batch_normalization_2')(x)
    
    xw = AveragePooling2D(name='average_pooling2d_1')(x)
    xw = Conv2D(256, (3, 3), padding='same', activation='relu')(xw)
    xw = Conv2D(4, (3, 3), padding='same')(xw)
    xw = GlobalAveragePooling2D()(xw)
    xw_output = Activation('softmax', name='weather')(xw)
    
    xnw = MaxPooling2D(name='max_pooling2d_1')(x)
    xnw = Conv2D(256, (3, 3), padding='same', activation='relu')(xnw)
    xnw = Conv2D(13, (3, 3), padding='same')(xnw)
    xnw = GlobalMaxPooling2D()(xnw)
    xnw_output = Activation('sigmoid', name='common_rare')(xnw)

    model = Model(inputs=inputs, outputs=[xw_output, xnw_output])
    model.name = "mini-VGG_2head_Adam"

    opt = Adam(lr=lr,
               decay=lr_decay)

    model.compile(loss=loss, optimizer=opt, metrics=['accuracy', f2_score])
    return model

#get_mini_vgg_2head((64, 64, 3)).summary()

def process_labels(train_df):
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = sorted(
        list(
            set(flatten([l.split(' ') for l in train_df['tags'].values]))
        )
    )

    label_map = {l: i for i, l in enumerate(labels)}
    inv_label_map = {i: l for l, i in label_map.items()}

    Y = []
    for f, tags in train_df.values:
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        Y.append(targets)

    Y = np.array(Y, np.uint8)
    return label_map, inv_label_map, Y

def stratified_kfold_sampling(Y, n_splits=10, num_classes=17, random_state=0):
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


# Savling/loading arrays by bcolz
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w', expectedlen=len(arr))
    c.flush()
    del c
    del arr
    
def load_array(fname, mode='r'):
    return bcolz.open(fname, mode=mode)[:]

def load_carray(fname, mode='r'):
    return bcolz.open(fname, mode=mode)



TRAIN_DATA = os.path.join(INPUT_PATH, "train")
train_jpeg_dir = os.path.join(INPUT_PATH, 'train', 'jpg')
test_jpeg_dir = os.path.join(INPUT_PATH, 'test', 'jpg')
train_csv_file = os.path.join(INPUT_PATH, 'train_v2.csv')
test_csv_file = os.path.join(INPUT_PATH, 'sample_submission_v2.csv')

####### Load data ######

labels_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

label_map, inv_label_map, Y = process_labels(labels_df)

#pd.DataFrame.from_dict({i: l for i, l in zip(labels_df['image_name'], Y)}, orient='index').head()



# Params
input_size = 96
input_channels = 3

epochs = 40
batch_size = 128
learning_rate = 0.001
lr_decay = 1e-4

n_folds = 5

train_folds, valid_folds = stratified_kfold_sampling(Y, n_splits= n_folds,
                                                     random_state=random_seed)

#all_steps = len(Y) // batch_size + 1 #np.ceil(len(Y)/batch_size)

#(1 - np.sum(Y, axis=0)/np.sum(Y))[2:9]
#
#np.sum(Y)/np.sum(Y, axis=0)
#
#np.mean(np.sum(Y)/np.sum(Y, axis=0))

#ml_dict = {os.path.join('jpg', '{}.jpg').format(i): l for i, l in zip(labels_df['image_name'], Y)}


gc.collect()

if os.path.isfile('train_array_{}'.format(input_size) + '.dat'):
    print("loading data in memory from bcolz file "+ 'train_array_{}'.format(input_size) + '.dat')
    train_array = load_array('train_array_{}'.format(input_size) + '.dat')
else:
    train_array =[]
    for f in tqdm(labels_df['image_name'].values, miniters=1000):
        img = cv2.resize(cv2.imread(
                os.path.join(TRAIN_DATA, 'jpg', '{}.jpg'.format(f))
                                   ), (input_size, input_size))
        train_array.append(img)
    
    train_array = np.array(train_array, np.float32)
    
    save_array('train_array_{}'.format(input_size) + '.dat', train_array)
    gc.collect()



#train_gen = image_ml_ext.ImageDataGenerator()
#valid_gen = image_ml_ext.ImageDataGenerator()

folds_to_use = [4]
class_count = 17

weather_labels = ['clear', 'cloudy', 'haze', 'partly_cloudy']
weather_labels_inx = [label_map.get(label) for label in weather_labels]
not_weather_labels_inx = [i for i in range(0, class_count) if i not in weather_labels_inx]


#[Y[0:5,weather_labels_inx], 
#     Y[0:5,not_weather_labels_inx]]
#

def getImageDataGenerator():
    return image_ml_ext.ImageDataGenerator(
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True,)

def getMOImageDataGenerator(X, Y, batch_size, shuffle=True):
    while 1:
        if shuffle:
            inx = np.random.permutation(X.shape[0])
        else:
            inx = range(0, X.shape[0])
        genX = getImageDataGenerator().flow(X[inx], None, batch_size=batch_size, shuffle=False)

        for i in range(0, len(inx), batch_size):
            yield genX.next(), [y[inx[i:i+batch_size]] for y in Y]
			
all_gen = getMOImageDataGenerator(
    train_array, 
    [Y[:,weather_labels_inx], 
     Y[:,not_weather_labels_inx]],
    batch_size, False)
        
#train_gen = getMOImageDataGenerator(
#    train_array[train_folds[fold_inx]], 
#    [Y[train_folds[fold_inx]][:,weather_labels_inx], 
#     Y[train_folds[fold_inx]][:,not_weather_labels_inx]],
#    batch_size)
#
#valid_gen = getMOImageDataGenerator(
#    train_array[valid_folds[fold_inx]], 
#    [Y[valid_folds[fold_inx]][:,weather_labels_inx], 
#     Y[valid_folds[fold_inx]][:,not_weather_labels_inx]],
#    batch_size, False)



#train_gen = getImageDataGenerator()
#valid_gen = getImageDataGenerator()

#tst = [Y[train_folds[fold_inx]][:,weather_labels_inx], 
#                    Y[train_folds[fold_inx]][:,not_weather_labels_inx]]
#
#fold_inx = 4
#
#tst = np.hstack((Y[train_folds[fold_inx]][:,weather_labels_inx], Y[train_folds[fold_inx]][:,not_weather_labels_inx]))
#
#
#tst = [Y[train_folds[fold_inx]][:,weather_labels_inx], 
#                    Y[train_folds[fold_inx]][:,not_weather_labels_inx]]


for fold_inx in folds_to_use:
    print('Fold index {}'.format(fold_inx))

#    cache_images(os.path.join(INPUT_PATH, "train_images_" + str(input_size) + ".dat"),
#                 labels_df['image_name'].values[train_folds[fold_inx]],
#                 train_jpeg_dir, True, input_size, input_size, dtype=np.float32)

    train_gen = getMOImageDataGenerator(
        train_array[train_folds[fold_inx]], 
        [Y[train_folds[fold_inx]][:,weather_labels_inx], 
         Y[train_folds[fold_inx]][:,not_weather_labels_inx]],
        batch_size)
    
    valid_gen = getMOImageDataGenerator(
        train_array[valid_folds[fold_inx]], 
        [Y[valid_folds[fold_inx]][:,weather_labels_inx], 
         Y[valid_folds[fold_inx]][:,not_weather_labels_inx]],
        batch_size, False)


    train_steps = len(train_folds[fold_inx]) // batch_size + 1
#    train_ml_dict = {os.path.join('jpg', '{}.jpg').format(i): l for i, l in \
#               zip(labels_df['image_name'].values[train_folds[fold_inx]], \
#                   [Y[train_folds[fold_inx]][:,weather_labels_inx], 
#                    Y[train_folds[fold_inx]][:,not_weather_labels_inx]]
##                    Y[train_folds[fold_inx]]
#                   )}
#    train_batches = train_gen.flow_from_directory(TRAIN_DATA,
#                                  class_mode = 'multilabel',
#                                  multilabel_classes = train_ml_dict,
#                                  n_class = class_count,
#                                  target_size = (input_size, input_size),
#                                  batch_size = batch_size,
#                                  seed = random_seed)
#    train_batches = train_gen.flow(
#            labels_df['image_name'].values[train_folds[fold_inx]], 
#            [Y[train_folds[fold_inx]][:,weather_labels_inx], 
#             Y[train_folds[fold_inx]][:,not_weather_labels_inx]],
#                                  batch_size = batch_size,
#                                  seed = random_seed)

    valid_steps = len(valid_folds[fold_inx]) // batch_size + 1
#    valid_ml_dict = {os.path.join('jpg', '{}.jpg').format(i): l for i, l in \
#               zip(labels_df['image_name'].values[valid_folds[fold_inx]], \
#                    [Y[valid_folds[fold_inx]][:,weather_labels_inx], 
#                     Y[valid_folds[fold_inx]][:,not_weather_labels_inx]],
##                   Y[valid_folds[fold_inx]]
#                   )}
#    valid_batches = valid_gen.flow_from_directory(TRAIN_DATA,
#                                  class_mode = 'multilabel',
#                                  multilabel_classes = valid_ml_dict,
#                                  n_class = class_count,
#                                  target_size = (input_size, input_size),
#                                  batch_size = batch_size,
#                                  seed = random_seed)

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=0),
                 TensorBoard(log_dir='logs'),
                 ModelCheckpoint(name + \
                     '{}weights'.format((input_size, input_size, input_channels),
                      class_count) + 'fold{}'.format(fold_inx) + '.h5',
                                 save_best_only=True)]
    minivgg = get_mini_vgg_2head((input_size, input_size, input_channels))
    if os.path.isfile(name + \
                     '{}weights'.format((input_size, input_size, input_channels),
                      class_count) + 'fold{}'.format(fold_inx) + '.h5'):
        minivgg.load_weights(name + \
                     '{}weights'.format((input_size, input_size, input_channels),
                      class_count) + 'fold{}'.format(fold_inx) + '.h5')
    minivgg.fit_generator(
#        train_batches,
        train_gen,
        steps_per_epoch = train_steps,
        epochs = epochs,
#        validation_data = valid_batches,
        validation_data = valid_gen,
        validation_steps = valid_steps,
        callbacks = callbacks,
        max_q_size = 2
    )



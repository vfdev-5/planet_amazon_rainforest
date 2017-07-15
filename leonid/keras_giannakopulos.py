# -*- coding: utf-8 -*-
## Based on https://www.kaggle.com/petrosgk/1st-try-with-keras-0-918-lb
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import time

import cv2
from tqdm import tqdm

import h5py # for saving later usable files

import gc
import os

from sklearn.metrics import fbeta_score

import random
random_seed = 2017
random.seed(random_seed)
np.random.seed(random_seed)

import tensorflow as tf
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

try:
    if __file__: exit
except NameError:
    __file__ = 'leonid/keras_giannakopulos.py'

INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input'))
FILE_FOLDER = os.path.abspath(os.path.dirname(__file__))


# Params
input_size = 64
input_channels = 3

epochs = 40
batch_size = 192
learning_rate = 0.001
lr_decay = 1e-4

valid_data_size = 5000  # Samples to withhold for validation

model = Sequential()
model.add(BatchNormalization(input_shape=(input_size, input_size, input_channels)))
model.add(Conv2D(32, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(256, kernel_size=(2, 2), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(17, activation='sigmoid'))

df_train_data = pd.read_csv(INPUT_PATH + '/train_v2.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train_data['tags'].values])))

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

df_test_data = pd.read_csv(INPUT_PATH + '/sample_submission_v2.csv')

if os.path.isfile(FILE_FOLDER + '/storage.h5'):
    print("loading data in memory from hdf5 file 'storage.h5'")
    with h5py.File(FILE_FOLDER + '/storage.h5', 'r') as hf:
        x_train = hf['x_train'][:]
        y_train = hf['y_train'][:]
        x_valid = hf['x_valid'][:]
        y_valid = hf['y_valid'][:]
        x_test = hf['x_test'][:]
else:
    x_valid = []
    y_valid = []
    
    df_valid = df_train_data[(len(df_train_data) - valid_data_size):]
    
    for f, tags in tqdm(df_valid.values, miniters=100):
        img = cv2.resize(cv2.imread(INPUT_PATH + '/train/jpg/{}.jpg'.format(f)), (input_size, input_size))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_valid.append(img)
        y_valid.append(targets)
    
    y_valid = np.array(y_valid, np.uint8)
    x_valid = np.array(x_valid, np.float32)
    
    x_train = []
    y_train = []
    
    df_train = df_train_data[:(len(df_train_data) - valid_data_size)]
    
    for f, tags in tqdm(df_train.values, miniters=1000):
        img = cv2.resize(cv2.imread(INPUT_PATH + '/train/jpg/{}.jpg'.format(f)), (input_size, input_size))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(img)
        y_train.append(targets)
        img = cv2.flip(img, 0)  # flip vertically
        x_train.append(img)
        y_train.append(targets)
        img = cv2.flip(img, 1)  # flip horizontally
        x_train.append(img)
        y_train.append(targets)
        img = cv2.flip(img, 0)  # flip vertically
        x_train.append(img)
        y_train.append(targets)
    
    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float32)
    
    x_test = []
    
    for f, tags in tqdm(df_test_data.values, miniters=1000):
        img = cv2.resize(cv2.imread(INPUT_PATH + '/test/jpg/{}.jpg'.format(f)), (input_size, input_size))
        x_test.append(img)
    
    x_test = np.array(x_test, np.float32)
    
    with h5py.File(FILE_FOLDER + '/storage.h5', 'w') as hf:
        hf.create_dataset("x_train",  data=x_train, compression="lzf")# compression="gzip", compression_opts=9
        hf.create_dataset("y_train",  data=y_train, compression="lzf")
        hf.create_dataset("x_valid",  data=x_valid, compression="lzf")
        hf.create_dataset("y_valid",  data=y_valid, compression="lzf")
        hf.create_dataset("x_test",  data=x_test, compression="lzf")

##del x_train
##del y_train
##del x_valid
##del y_valid
##del x_test

gc.collect()

#if os.path.isfile('weights64.h5'):
#    print("loading existing weight for training")
#    model.load_weights('weights64.h5')

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=5,
                           verbose=0),
             TensorBoard(log_dir='logs'),
             ModelCheckpoint(FILE_FOLDER + '/weights64.h5',
                             save_best_only=True)]

opt = Adam(lr=learning_rate, decay=lr_decay)

model.compile(loss='binary_crossentropy',
              # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          callbacks=callbacks,
          validation_data=(x_valid, y_valid))

if os.path.isfile(FILE_FOLDER + '/weights64.h5'):
    model.load_weights(FILE_FOLDER + '/weights64.h5')
    
p_valid = model.predict(x_valid, batch_size=batch_size)
print("Validation f2 score (threshold 0.2): ", fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

#Epoch 21/40
#59s - loss: 0.0861 - acc: 0.9662 - val_loss: 0.0939 - val_acc: 0.9642
#Epoch 22/40
#60s - loss: 0.0854 - acc: 0.9665 - val_loss: 0.0962 - val_acc: 0.9638
#Epoch 23/40
#60s - loss: 0.0844 - acc: 0.9667 - val_loss: 0.0949 - val_acc: 0.9636
#Validation f2 score:  0.920758834062
#Validation f2 score (threshold 0.2):  0.91972756761

################################

def fbeta(true_label, prediction):
   return fbeta_score(true_label, prediction, beta=2, average='samples')

def get_optimal_threshhold(true_label, prediction, iterations = 100):
    best_threshhold = [0.2]*17    
    for t in range(17):
        best_fbeta = 0
        temp_threshhold = [0.2]*17
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = fbeta(true_label, prediction > temp_threshhold)
            if  temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    return best_threshhold

start = time.time()
thresholds = get_optimal_threshhold(y_valid, p_valid)
print(time.time() - start)

fbeta(y_valid, np.array(np.array(p_valid) > thresholds, np.uint8))

##############################################

def search_best_threshold(y_true, y_preds):
    from common.metrics import score
    _thr = 0.15    
    _bs = score(y_true, y_preds > _thr)
    
    thrs = np.arange(0.0, 1.0, 0.01)
    for thr in thrs:
        s = score(y_true, y_preds > thr)
        if s > _bs:
            _bs = s
            _thr = thr      
    return _thr, _bs

best_thresholds = {}

for tag_index, tag in inv_label_map.items():
    best_thresholds[tag], best_score = \
    search_best_threshold(y_valid[:, tag_index],
                          p_valid[:, tag_index])
    print("%s | best threshold : %f with score: %f" % (tag, best_thresholds[tag], best_score))

#road | best threshold : 0.070000 with score: 0.822785
#selective_logging | best threshold : 0.060000 with score: 0.380435
#agriculture | best threshold : 0.170000 with score: 0.774494
#haze | best threshold : 0.050000 with score: 0.375000
#primary | best threshold : 0.430000 with score: 0.609756
#clear | best threshold : 0.160000 with score: 0.895441
#blooming | best threshold : 0.100000 with score: 0.940999
#artisinal_mine | best threshold : 0.060000 with score: 0.313283
#blow_down | best threshold : 0.200000 with score: 0.839416
#slash_burn | best threshold : 0.130000 with score: 0.699401
#partly_cloudy | best threshold : 0.030000 with score: 0.219561
#bare_ground | best threshold : 0.200000 with score: 0.873328
#cultivation | best threshold : 0.170000 with score: 0.741815
#habitation | best threshold : 0.070000 with score: 0.476493
#cloudy | best threshold : 0.340000 with score: 0.990112
#water | best threshold : 0.290000 with score: 0.977382
#conventional_mine | best threshold : 0.150000 with score: 0.791762
##############################################

print("Validation f2 score (thresholds): ",
      fbeta_score(y_valid,
                  np.array(p_valid) > thresholds,
                  beta=2, average='samples'))
#Validation f2 score (thresholds):  0.922405779889






y_test = []

p_test = model.predict(x_test, batch_size=batch_size, verbose=1)
y_test.append(p_test)

result = np.array(y_test[0])
result = pd.DataFrame(result, columns=labels)

result.to_csv(FILE_FOLDER + '/keras_giannakopulos_result_pred.csv',
                        index=False)

##################################
#Model results on full train data
x_fulltrain = []
y_fulltrain = []

for f, tags in tqdm(df_train_data.values, miniters=100):
    img = cv2.resize(
            cv2.imread(INPUT_PATH + '/train/jpg/{}.jpg'.format(f)),
            (input_size, input_size))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_fulltrain.append(img)
    y_fulltrain.append(targets)

y_fulltrain = np.array(y_fulltrain, np.uint8)
x_fulltrain = np.array(x_fulltrain, np.float32)

p_fulltrain = model.predict(x_fulltrain, batch_size=batch_size, verbose=1)
print("Full train set f2 score (threshold 0.2): ",
      fbeta_score(y_fulltrain, np.array(p_fulltrain) > 0.2,
                  beta=2, average='samples'))
print("Full train set f2 score (thresholds): ",
      fbeta_score(y_fulltrain, np.array(p_fulltrain) > thresholds,
                  beta=2, average='samples'))

result_fulltrain = pd.DataFrame(np.array(p_fulltrain), columns=labels)
result_fulltrain.to_csv(FILE_FOLDER + '/keras_giannakopulos_fulltrain_pred.csv',
                        index=False)

fulltrain_preds = []
for i in tqdm(range(result_fulltrain.shape[0]), miniters=1000):
    a = result_fulltrain.iloc[[i]]
    a = a.apply(lambda x: x > thresholds, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    fulltrain_preds.append(' '.join(list(a.index)))

fulltrain_submit = df_train_data.copy()
fulltrain_submit['tags'] = fulltrain_preds
fulltrain_submit.to_csv(FILE_FOLDER + '/keras_giannakopulos_fulltrain_submit.csv',
                        index=False)

##################################



preds = []

for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.iloc[[i]]
    a = a.apply(lambda x: x > thresholds, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))

df_test_data['tags'] = preds
df_test_data.to_csv(
        os.path.join(FILE_FOLDER,
             'keras_giannakopulos_submission' \
             + '_{:.5}'.format(
                fbeta_score(y_valid,
                        np.array(p_valid) > thresholds, beta=2, average='samples')) \
             + '.csv')
        , index=False)

#Validation f2 score (thresholds):  0.922405779889


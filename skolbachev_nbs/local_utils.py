from time import time
t_start_import = time()

# General imports
import os, sys
from joblib import Parallel, delayed
from multiprocessing import Pool
from functools import partial
from glob import glob
import bcolz
from IPython.display import SVG
from IPython.lib.display import FileLink

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

WORK_DIR = os.path.abspath('..')
NB_DIR = WORK_DIR + '/planet_understanding_the_amazon_from_space'
DATA_DIR = NB_DIR + '/data'
BATCH_SIZE = 32

# Scikit-learn imports
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

# Keras imports
from keras import backend as K
from keras import optimizers
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, concatenate
from keras.layers.core import Activation
from keras.callbacks import Callback, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras_metrics import *

# Importing pretrained models
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
sys.path.insert(0, WORK_DIR + '/keras_additional_models')
from squeezenet import SqueezeNet

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


# Image loading
def load_image(file_name, folder, resize=False, r_x=64, r_y=64, dtype=np.float32):
    img = cv2.imread(folder + '/{}.jpg'.format(file_name))
    if resize:
        img = cv2.resize(img, (r_x, r_y))
    return np.array(img, dtype) / 255.

def load_images(file_names, folder, resize=False, r_x=64, r_y=64, dtype=np.float32, cpu_cores=8):
    return np.array(Parallel(n_jobs=cpu_cores)(delayed(load_image)(file_name, folder, resize, r_x, r_y, dtype) for file_name in file_names))

# TODO
def load_images_generator(batch_size, file_names, folder, resize=False, r_x=64, r_y=64, dtype=np.float32):
    while 1:
        for i in range(0, len(file_names), batch_size):
            yield load_images(file_names[i:i+batch_size], folder, resize, r_x, r_y, dtype)

            
# Cache images arrays on disk
def cache_images(cache_fine_name, file_names, folder, resize=False, r_x=64, r_y=64, dtype=np.float32):
    save_array(cache_fine_name, load_images(file_names, folder, resize, r_x, r_y, dtype))
    

# Data array generator
def data_fit_generator(X, Y, inx, batch_size, shuffle=False):
    if shuffle:
        np.random.shuffle(inx)
    
    while 1:
        for i in range(0, len(inx), batch_size):
            yield X[inx[i:i+batch_size]], Y[inx[i:i+batch_size]]
            
def data_predict_generator(X, size, batch_size):
    while 1:
        for i in range(0, size, batch_size):
            yield X[i:i+batch_size]
    

# Getting pretrained model       
def get_pretrained_model(model_name, input_shape, include_top=False, weights="imagenet", pooling=None):
    if model_name == "vgg16":
        return VGG16(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
    if model_name == "vgg19":
        return VGG19(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
    if model_name == "resnet50":
        return ResNet50(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
    if model_name == "inceptionV3":
        return InceptionV3(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
    if model_name == "xception":
        return Xception(include_top=include_top, weights=weights, input_shape=input_shape, pooling=pooling)
    if model_name == "squeezenet":
        return SqueezeNet(include_top=include_top, weights=weights, input_shape=input_shape)

    
# Stratified (KFold) Sampling
def stratified_sampling(Y, split=0.1, num_classes=17, random_state=0):
    train_inx = []
    valid_inx = []
    inx = np.arange(len(Y))

    for i in range(0,num_classes):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=split, random_state=random_state+i)
        b_train_inx, b_valid_inx = next(sss.split(inx, Y[:,i]))
        # to ensure there is no repetetion within each split and between the splits
        train_inx = train_inx + list(set(list(b_train_inx)) - set(train_inx) - set(valid_inx))
        valid_inx = valid_inx + list(set(list(b_valid_inx)) - set(train_inx) - set(valid_inx))
        
    return train_inx, valid_inx

def stratified_kfold_sampling(Y, n_splits=10, num_classes=17, random_state=0):
    train_folds = [[] for _ in range(n_splits)]
    valid_folds = [[] for _ in range(n_splits)]
    inx = np.arange(len(Y))
    valid_size = 1.0 / n_splits

    for cl in range(0, num_classes):
        sss = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state+cl)
        
        for fold, (train_index, test_index) in enumerate(sss.split(inx, Y[:,cl])):
            b_train_inx, b_valid_inx = inx[train_index], inx[test_index]
            
            # to ensure there is no repetetion within each split and between the splits
            train_folds[fold] = train_folds[fold] + list(set(list(b_train_inx)) - set(train_folds[fold]) - set(valid_folds[fold]))
            valid_folds[fold] = valid_folds[fold] + list(set(list(b_valid_inx)) - set(train_folds[fold]) - set(valid_folds[fold]))
        
    return np.array(train_folds), np.array(valid_folds)


# Labels processing
def process_labels(train_df):
    flatten = lambda l: [item for sublist in l for item in sublist]
    labels = sorted(list(set(flatten([l.split(' ') for l in train_df['tags'].values]))))

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


def predictTTA(model, img_array, img_gen, steps, batch_size, numTTA):
    p_no_tta = model.predict(img_array, batch_size=batch_size, verbose=1)
    
    p_tta_list = []
    
    for i in range(0, numTTA):
        print("Predicting TTA: {}".format(i))
        p_tta = model.predict_generator(
            img_gen.flow(img_array, None, batch_size=batch_size, shuffle=False),
            steps=steps, verbose=1)
        p_tta_list.append(p_tta)
        
    return p_no_tta, p_tta_list

# F2 Score + finding best one via bruteforce
def f2_score(Y, p_X, thres):
    return fbeta_score(Y, p_X >= thres, beta=2, average='samples')

class F2History(Callback):
    def __init__(self, img_array, Y, img_gen, steps, batch_size):
        self.img_array = img_array
        self.img_gen = img_gen
        self.steps = steps
        self.batch_size = batch_size
        self.Y = Y
        self.f2_02_scores = []
        self.f2_05_scores = []
    
    def on_epoch_end(self, epoch, logs ={}):
        p_X = self.model.predict_generator(
            self.img_gen.flow(self.img_array, None, batch_size=self.batch_size, shuffle=False), 
            self.steps)
        f2_02 = f2_score(self.Y, np.array(p_X), 0.2)
        f2_05 = f2_score(self.Y, np.array(p_X), 0.5)
        print()
        print("F2 Score (0.2) {}; (0.5) {};".format(f2_02, f2_05))
        self.f2_02_scores.append(f2_02)
        self.f2_05_scores.append(f2_05)
        return

def find_threshold(Y, p_X, label_inx, grid=np.arange(0.0, 1.0, 0.0005)):
    thres = np.full((17,), 0.2)
    thres[label_inx] = 0.0
    max_th = 0.0
    max_score = f2_score(Y, p_X, thres)

    for th in grid:
        thres[label_inx] = th
        score = f2_score(Y, p_X, thres)
        if score > max_score:
            max_score = score
            max_th = th

    print("{}: Score {} with {}".format(label_inx, max_score, max_th))
    return (label_inx, max_th)
    
def find_best_thresholds(Y, p_X, grid=np.arange(0.0, 1.0, 0.0005), cpu_cores=8):
    thres_map = dict(Parallel(n_jobs=cpu_cores)(delayed(find_threshold)(Y, p_X, label_inx, grid) 
                                                for label_inx in np.arange(0,17)))
    return [thres_map[i] for i in range(0,17)]

print("Import time: {}".format(time() - t_start_import))
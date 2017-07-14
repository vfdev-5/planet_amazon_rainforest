from time import time
t_start_import = time()

import os, sys
from joblib import Parallel, delayed
from glob import glob
import bcolz
from IPython.lib.display import FileLink
# sys.path.insert(0, '/src/utils')

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, BatchNormalization, concatenate
from keras.layers.core import Activation
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras import backend as K


NB_DIR = os.getcwd()
DATA_DIR = NB_DIR + '/data'
BATCH_SIZE = 32

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()
    
def load_array(fname):
    return bcolz.open(fname)[:]

def load_carray(fname, mode='r'):
    return bcolz.open(fname, mode=mode)

def load_image(file_name, folder, resize=False, r_x=64, r_y=64, dtype=np.float16):
    img = cv2.imread(folder + '/{}.jpg'.format(file_name))
    if resize:
        img = cv2.resize(img, (r_x, r_y))
    return np.array(img, dtype) / 255.

def load_images(file_names, folder, resize=False, r_x=64, r_y=64, dtype=np.float16, cpu_cores=8):
    return np.array(Parallel(n_jobs=cpu_cores)(delayed(load_image)(file_name, folder, resize, r_x, r_y, dtype) 
                                               for file_name in file_names))

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

def f2_score(Y, p_X, thres):
    return fbeta_score(Y, p_X >= thres, beta=2, average='samples')

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
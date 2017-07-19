# # Train finetunned SqueezeNet 2 model for multi-label classification


import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

try:
    if __file__: exit
except NameError:
    __file__ = 'leonid/train_mini_vgg_mlc_ac.py'

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)
if not os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'skolbachev_nbs')) in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'skolbachev_nbs')))

import numpy as np

from data_utils import get_id_type_list_for_class, OUTPUT_PATH, GENERATED_DATA, to_set
from training_utils import classification_train as train, classification_validate as validate
from training_utils import exp_decay, step_decay, get_basic_imgaug_seq

from models.mini_vgg_multiclassification import get_mini_vgg_bn, get_mini_vgg

from sklearn.model_selection import KFold
from data_utils import to_set, equalized_data_classes, unique_tags, train_jpg_ids, TRAIN_ENC_CL_CSV
from data_utils import load_pretrained_model, get_label
from data_utils import DataCache

from xy_providers import image_label_provider
from models.keras_metrics import binary_crossentropy_with_false_negatives

#from models.loss_lr_scheduler import LossLearningRateScheduler

#from loss_lr_scheduler import LossLearningRateScheduler


cnn = get_mini_vgg(input_shape=(64, 64, 3), n_classes=17)
cnn.summary()

# Setup configuration

seed = 2017
np.random.seed(seed)

trainval_id_type_list = [(image_id, "Train_jpg") for image_id in train_jpg_ids]
np.random.shuffle(trainval_id_type_list)
print(len(trainval_id_type_list))

cache = DataCache(30000)  # !!! CHECK BEFORE LOAD TO FLOYD

#loss_fun = LossLearningRateScheduler(base_lr = 0.001, lookback_epochs = 3)

params = {
    'seed': seed,

    'xy_provider': image_label_provider,

    'network': get_mini_vgg,
    'optimizer': 'adam',
    'loss': 'binary_crossentropy',# 'binary_crossentropy', # mae_with_false_negatives,
    'nb_epochs': 1,    # !!! CHECK BEFORE LOAD TO FLOYD
    'batch_size': 64,  # !!! CHECK BEFORE LOAD TO FLOYD#tensor with shape[batch_size,32,31,31] 1024 for (64,64) image and 8G video RAM is good


    'normalize_data': False,
    'normalization': '',

    'image_size': (128, 128),

    # Learning rate scheduler
    'lr_kwargs': {
        'lr': 0.01,
        'a': 0.95,
        'init_epoch': 0
    },
    'lr_decay_f': exp_decay,

    'train_seq': get_basic_imgaug_seq,

    # Reduce learning rate on plateau
    'on_plateau': True,
    'on_plateau_kwargs': {
        'monitor': 'val_loss',
        'factor': 0.1,
        'patience': 3,
        'verbose': 1
    },

    'cache': cache,

    # 'class_index': 0,
     'pretrained_model': 'load_best',
    # 'pretrained_model': os.path.join(GENERATED_DATA, "weights", ""),

    'output_path': OUTPUT_PATH,
}

params['save_prefix_template'] = '{cnn_name}_all_classes_fold={fold_index}_seed=%i' % params['seed']
params['input_shape'] = params['image_size'] + (3,)
params['n_classes'] = len(unique_tags)

#params['imgaug_seq'] = None
# Start CV

n_folds = 5
val_fold_index = 0
val_fold_indices = []  # !!! CHECK BEFORE LOAD TO FLOYD
hists = []

###########################################

from sklearn.model_selection import StratifiedKFold
def stratified_kfold_sampling(Y, n_splits=5, num_classes=17, random_state=2017):
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

#    return train_folds, valid_folds
    return np.array(train_folds), np.array(valid_folds)



INPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'input'))
train_df = pd.read_csv(INPUT_PATH + '/train_v2.csv')
from local_utils import process_labels
label_map, inv_label_map, Y = process_labels(train_df)

########################################


#kf = KFold(n_splits=n_folds)
trainval_id_type_list = np.array(trainval_id_type_list)
#for train_index, test_index in kf.split(trainval_id_type_list):
train_folds, valid_folds = stratified_kfold_sampling(Y, n_splits=n_folds)
for train_index, test_index in zip(train_folds, valid_folds):
#    print("TRAIN:", train_index, "TEST:", test_index)
    train_id_type_list, val_id_type_list = trainval_id_type_list[train_index], trainval_id_type_list[test_index]

    if len(val_fold_indices) > 0:
        if val_fold_index not in val_fold_indices:
            val_fold_index += 1
            continue

    val_fold_index += 1
    print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)

    print(datetime.now(), len(train_id_type_list), len(val_id_type_list))
    assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"

    weights = None if 'pretrained_model' in params else None
    cnn = params['network'](lr=params['lr_kwargs']['lr'], weights=weights, **params)
    params['save_prefix'] = params['save_prefix_template'].format(cnn_name=cnn.name, fold_index=val_fold_index - 1)
    print("\n {} - Loaded {} model ...".format(datetime.now(), cnn.name))

    if 'pretrained_model' in params:
        load_pretrained_model(cnn, **params)

    print("\n {} - Start training ...".format(datetime.now()))
    h = train(cnn, train_id_type_list, val_id_type_list, **params)
    if h is None:
        continue
    hists.append(h)


# ### Validation all classes

n_runs = 2
n_folds = 5
run_counter = 0
cv_mean_scores = np.zeros((n_runs, n_folds))
val_fold_indices = []  # !!! CHECK BEFORE LOAD TO FLOYD

params['pretrained_model'] = 'load_best'

_trainval_id_type_list = np.array(trainval_id_type_list)

while run_counter < n_runs:
    run_counter += 1
    print("\n\n ---- New run : ", run_counter, "/", n_runs)
    val_fold_index = 0
#    kf = KFold(n_splits=n_folds)
#    for train_index, test_index in kf.split(_trainval_id_type_list):
    train_folds, valid_folds = stratified_kfold_sampling(Y, n_splits=n_folds)
    for train_index, test_index in zip(train_folds, valid_folds):
#        print("TRAIN:", train_index, "TEST:", test_index)
        train_id_type_list, val_id_type_list = _trainval_id_type_list[train_index], _trainval_id_type_list[test_index]

        if len(val_fold_indices) > 0:
            if val_fold_index not in val_fold_indices:
                val_fold_index += 1
                continue

        val_fold_index += 1
        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)

        print(len(train_id_type_list), len(val_id_type_list))
        assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"

        cnn = params['network'](input_shape=params['input_shape'], n_classes=params['n_classes'])
        params['save_prefix'] = params['save_prefix_template'].format(cnn_name=cnn.name, fold_index=val_fold_index-1)
        print("\n {} - Loaded {} model ...".format(datetime.now(), cnn.name))

        load_pretrained_model(cnn, **params)

        params['seed'] += run_counter - 1

        f2, mae = validate(cnn, val_id_type_list, verbose=0, **params)
        cv_mean_scores[run_counter-1, val_fold_index-1] = f2

    np.random.shuffle(_trainval_id_type_list)

print(cv_mean_scores)

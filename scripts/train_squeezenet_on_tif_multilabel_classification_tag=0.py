# # Train finetunned SqueezeNet 2 model for multi-label classification


import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    if __file__: exit
except NameError:
    __file__ = 'scripts/train_squeezenet21_multilabel_classification_all_classes.py'

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

import numpy as np

from data_utils import get_id_type_list_for_class, OUTPUT_PATH, GENERATED_DATA, to_set
from training_utils import classification_train as train, classification_validate as validate
from training_utils import exp_decay, step_decay

from models.squeezenet_tif_classification import get_squeezenet_on_tif

from sklearn.model_selection import KFold
from data_utils import to_set, equalized_data_classes, unique_tags, train_jpg_ids, TRAIN_ENC_CL_CSV
from data_utils import load_pretrained_model, get_label
from data_utils import DataCache

from xy_providers import tif_image_label_provider
from models.keras_metrics import binary_crossentropy_with_false_negatives


cnn = get_squeezenet_on_tif((224, 224, 7), len(equalized_data_classes[0]) + 1)
cnn.summary()

# Setup configuration

seed = 2017
np.random.seed(seed)

cache = DataCache(10000)  # !!! CHECK BEFORE LOAD TO FLOYD

class_index = 0
trainval_id_type_list = get_id_type_list_for_class(class_index, 'Train_tif')

### ADD GENERATED IMAGES
from glob import glob
from data_utils import GENERATED_DATA, to_set, get_label

gen_train_files = glob(os.path.join(GENERATED_DATA, "train", "tif", "*.tif"))
gen_train_ids = [s[len(os.path.join(GENERATED_DATA, "train", "tif"))+1+len('gen_train_'):-4] for s in gen_train_files]
gen_id_type_list = [(image_id, "Generated_Train_tif") for image_id in gen_train_ids]
class_index_gen_train_ids = [id_type for id_type in gen_train_ids if np.sum(get_label(*gen_id_type_list[0], class_index=class_index)) > 0]
class_index_gen_id_type_list = [(image_id, "Generated_Train_tif") for image_id in class_index_gen_train_ids]
trainval_id_type_list = trainval_id_type_list + class_index_gen_id_type_list
### ADD GENERATED IMAGES


class_indices = list(equalized_data_classes.keys())
class_indices.remove(class_index)

n_other_samples = int(len(trainval_id_type_list) * 1.0 / len(class_indices) / len(equalized_data_classes[class_index]))

for index in class_indices:
    id_type_list = np.array(get_id_type_list_for_class(index, 'Train_tif'))
    id_type_list = list(to_set(id_type_list) - to_set(trainval_id_type_list))
    np.random.shuffle(id_type_list)
    trainval_id_type_list.extend(id_type_list[:n_other_samples])


params = {
    'seed': seed,

    'xy_provider': tif_image_label_provider,

    'network': get_squeezenet_on_tif,
    'optimizer': 'adadelta',
    'loss': binary_crossentropy_with_false_negatives,
    'nb_epochs': 25,    # !!! CHECK BEFORE LOAD TO FLOYD
    'batch_size': 24,  # !!! CHECK BEFORE LOAD TO FLOYD

    'normalize_data': True,
    'normalization': '',

    'image_size': (256, 256),

    'class_index': class_index,

    # Learning rate scheduler
    'lr_kwargs': {
        'lr': 1.0,
        'a': 0.95,
        'init_epoch': 0
    },
    'lr_decay_f': exp_decay,

    # Reduce learning rate on plateau
    'on_plateau': True,
    'on_plateau_kwargs': {
        'monitor': 'val_loss',
        'factor': 0.1,
        'patience': 3,
        'verbose': 1
    },

    'cache': cache,

#      'pretrained_model': 'load_best',
#     'pretrained_model': os.path.join(GENERATED_DATA, "weights", ""),

    'output_path': OUTPUT_PATH,
}

params['save_prefix_template'] = '{cnn_name}_tif_class=%i_fold={fold_index}_seed=%i' % (class_index, params['seed'])
params['input_shape'] = params['image_size'] + (7,)
params['n_classes'] = len(equalized_data_classes[class_index]) + 1


# Start CV

n_folds = 5
val_fold_index = 0
val_fold_indices = []  # !!! CHECK BEFORE LOAD TO FLOYD
hists = []

kf = KFold(n_splits=n_folds)
trainval_id_type_list = np.array(trainval_id_type_list)
for train_index, test_index in kf.split(trainval_id_type_list):
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
    kf = KFold(n_splits=n_folds)
    for train_index, test_index in kf.split(_trainval_id_type_list):
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

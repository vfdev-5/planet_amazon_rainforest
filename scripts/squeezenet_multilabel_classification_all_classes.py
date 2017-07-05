# Train finetunned SqueezeNet model for multi-label classification


import os
import sys
from glob import glob
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

import numpy as np

from data_utils import get_id_type_list_for_class, OUTPUT_PATH, GENERATED_DATA, to_set
from training_utils import classification_train as train, classification_validate as validate
from training_utils import exp_decay, step_decay

from models.squeezenet_multiclassification import get_squeezenet

from sklearn.model_selection import KFold
from data_utils import to_set, equalized_data_classes, unique_tags, train_jpg_ids, TRAIN_ENC_CL_CSV
from data_utils import find_best_weights_file, get_label


from xy_providers import image_label_provider


cnn = get_squeezenet((256, 256, 3), 17)
cnn.summary()

# ## Train on all classes

seed = 2017
np.random.seed(seed)


# In[8]:


trainval_id_type_list = [(image_id, "Train_jpg") for image_id in train_jpg_ids]

print(len(trainval_id_type_list))


from data_utils import DataCache

try:
    if cache is None:
        cache_filepath = os.path.join(GENERATED_DATA, 'data_cache.pkl')
        if os.path.exists(cache_filepath):
            print("Load cache from pickle file")
            cache = load_data_cache(cache_filepath)
        else:
            cache = DataCache(0)
except NameError:
    cache_filepath = os.path.join(GENERATED_DATA, 'data_cache.pkl')
    if os.path.exists(cache_filepath):
        print("Load cache from pickle file")
        cache = load_data_cache(cache_filepath)
    else:
        cache = DataCache(0)


params = {
    'seed': seed,

    'xy_provider': image_label_provider,

    'network': get_squeezenet,
    'optimizer': 'adadelta',
    'loss': 'categorical_crossentropy',
    'nb_epochs': 100,
    'batch_size': 128,

    'normalize_data': True,
    'normalization': 'vgg',

    'image_size': (256, 256),

    'lr_kwargs': {
        'lr': 0.01,
        'a': 0.95,
        'init_epoch': 0
    },
    'lr_decay_f': exp_decay,

    'cache': cache,

    'class_index': 0,

#     'pretrained_model': 'load_best',
#     'pretrained_model': os.path.join(GENERATED_DATA, "weights", ""),

    'output_path': OUTPUT_PATH,


}

params['save_prefix_template'] = '{cnn_name}_all_classes_fold={fold_index}_seed=%i' % params['seed']
params['input_shape'] = params['image_size'] + (3,)
params['n_classes'] = len(unique_tags)


# Start CV

n_folds = 5
val_fold_index = 0
val_fold_indices = []
hists = []

kf = KFold(n_splits=n_folds, shuffle=True, random_state=params['seed'])
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

    cnn = params['network'](lr=params['lr_kwargs']['lr'], **params)
    params['save_prefix'] = params['save_prefix_template'].format(cnn_name=cnn.name, fold_index=val_fold_index-1)
    print("\n {} - Loaded {} model ...".format(datetime.now(), cnn.name))

    if 'pretrained_model' in params:
        if params['pretrained_model'] == 'load_best':
            weights_files = glob(os.path.join(params['output_path'], "weights", "%s*.h5" % params['save_prefix']))
            assert len(weights_files) > 0, "Failed to load weights"
            best_weights_filename, best_val_loss = find_best_weights_file(weights_files, field_name='val_loss')
            print("Load best loss weights: ", best_weights_filename, best_val_loss)
            cnn.load_weights(best_weights_filename)
        else:
            assert os.path.exist(params['pretrained_model']), "Not found pretrained model"
            print("Load weights: ", params['pretrained_model'])
            cnn.load_weights(params['pretrained_model'], by_name=True)

    print("\n {} - Start training ...".format(datetime.now()))
    h = train(cnn, train_id_type_list, val_id_type_list, **params)
    if h is None:
        continue
    hists.append(h)


# ### Validation all classes

n_runs = 1
n_folds = 5
run_counter = 0
cv_mean_scores = np.zeros((n_runs, n_folds))
val_fold_indices = []

while run_counter < n_runs:
    run_counter += 1
    print("\n\n ---- New run : ", run_counter, "/", n_runs)
    val_fold_index = 0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=params['seed'])
    trainval_id_type_list = np.array(trainval_id_type_list)
    for train_index, test_index in kf.split(trainval_id_type_list):
        train_id_type_list, val_id_type_list = trainval_id_type_list[train_index], trainval_id_type_list[test_index]

        if len(val_fold_indices) > 0:
            if val_fold_index not in val_fold_indices:
                val_fold_index += 1
                continue

        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)
        val_fold_index += 1

        print(len(train_id_type_list), len(val_id_type_list))
        assert len(to_set(train_id_type_list) & to_set(val_id_type_list)) == 0, "WTF"

        cnn = params['network'](input_shape=params['input_shape'], n_classes=params['n_classes'])
        params['save_prefix'] = params['save_prefix_template'].format(cnn_name=cnn.name, fold_index=val_fold_index-1)
        print("\n {} - Loaded {} model ...".format(datetime.now(), cnn.name))

        weights_files = glob(os.path.join(OUTPUT_PATH, "weights", "%s*.h5" % params['save_prefix']))
        assert len(weights_files) > 0, "Failed to load weights"
        best_weights_filename, best_val_loss = find_best_weights_file(weights_files, field_name='val_loss')
        print("Load best loss weights: ", best_weights_filename, best_val_loss)
        cnn.load_weights(best_weights_filename)

        score = validate(cnn, val_id_type_list, **params)
        cv_mean_scores[run_counter-1, val_fold_index-1] = score

print(cv_mean_scores)

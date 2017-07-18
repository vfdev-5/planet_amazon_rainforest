# # Train finetunned SqueezeNet 2 model for multi-label classification


import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    if __file__: exit
except NameError:
    __file__ = 'scripts/predict_squeezenet21_multilabel_classification_all_classes.py'

# Project
project_common_path = os.path.dirname(__file__)
project_common_path = os.path.abspath(os.path.join(project_common_path, '..', 'common'))
if not project_common_path in sys.path:
    sys.path.append(project_common_path)

import numpy as np

from data_utils import get_id_type_list_for_class, OUTPUT_PATH, GENERATED_DATA, to_set
from training_utils import classification_train as train, classification_validate as validate
from training_utils import exp_decay, step_decay

from models.squeezenet_multiclassification import get_squeezenet21

from sklearn.model_selection import KFold
from data_utils import to_set, equalized_data_classes, unique_tags, train_jpg_ids, TRAIN_ENC_CL_CSV
from data_utils import load_pretrained_model, get_label
from data_utils import DataCache

from xy_providers import image_label_provider
from models.keras_metrics import binary_crossentropy_with_false_negatives

from data_utils import test_jpg_ids, test_jpg_additional_ids
from test_utils import classification_predict as predict


# Setup configuration

seed = 2017
np.random.seed(seed)

cache = DataCache(10000)  # !!! CHECK BEFORE LOAD TO FLOYD

params = {
    'seed': seed,

    'xy_provider': image_label_provider,

    'network': get_squeezenet21,
    'optimizer': 'adadelta',
    'loss': binary_crossentropy_with_false_negatives, # 'binary_crossentropy', # mae_with_false_negatives,
    'nb_epochs': 25,    # !!! CHECK BEFORE LOAD TO FLOYD
    'batch_size': 128,  # !!! CHECK BEFORE LOAD TO FLOYD

    'normalize_data': True,
    'normalization': 'vgg',

    'image_size': (256, 256),

    # Learning rate scheduler
    'lr_kwargs': {
        'lr': 0.01,
        'a': 0.95,
        'init_epoch': 4
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

#     'class_index': 0,
    'pretrained_model': 'load_best',
#     'pretrained_model': os.path.join(GENERATED_DATA, "weights", ""),

    'output_path': OUTPUT_PATH,
}

params['save_prefix_template'] = '{cnn_name}_all_classes_fold={fold_index}_seed=%i' % params['seed']
params['input_shape'] = params['image_size'] + (3,)
params['n_classes'] = len(unique_tags)

test_id_type_list = []
test_id_type_list1 = []
for image_id in test_jpg_ids:
    test_id_type_list1.append((image_id, "Test_jpg"))

test_id_type_list2 = []
for image_id in test_jpg_additional_ids:
    test_id_type_list2.append((image_id, "ATest_jpg"))

test_id_type_list.extend(test_id_type_list1)
test_id_type_list.extend(test_id_type_list2)
print(len(test_id_type_list))

n_folds = 3  ## !!! CHECK THIS
run_counter = 0
n_runs = 2
params['pretrained_model'] = 'load_best'
now = datetime.now()

while run_counter < n_runs:
    run_counter += 1
    print("\n\n ---- New run : ", run_counter, "/", n_runs)

    # SqueezeNet on 5 folds
    for val_fold_index in range(n_folds):
        val_fold_index += 1
        print("\n\n ---- Validation fold index: ", val_fold_index, "/", n_folds)

        cnn = params['network'](input_shape=params['input_shape'], n_classes=params['n_classes'])
        params['save_prefix'] = params['save_prefix_template'].format(cnn_name=cnn.name, fold_index=val_fold_index - 1)
        print("\n {} - Loaded {} model ...".format(datetime.now(), cnn.name))

        load_pretrained_model(cnn, **params)

        params['seed'] += run_counter - 1
        df = predict(cnn, test_id_type_list, **params)
        info = params['save_prefix']
        sub_file = 'predictions_%i_%i_' % (run_counter, val_fold_index) + info + "_" + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
        sub_file = os.path.join(OUTPUT_PATH, sub_file)
        df.to_csv(sub_file, index=False)

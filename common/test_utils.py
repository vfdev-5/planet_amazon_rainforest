
import os
from datetime import datetime

import numpy as np
import pandas as pd


# Project
from data_utils import unique_tags
from training_utils import get_val_imgaug_seq, get_gen_flow

# ###### Classification #######


def classification_predict(model,
                           test_id_type_list,
                           **params):

    assert 'seed' in params, "Need seed, params = {}".format(params)
    verbose = 1 if 'verbose' not in params else params['verbose']

    val_seq = get_val_imgaug_seq(params['seed'])
    val_gen, val_flow = get_gen_flow(id_type_list=test_id_type_list,
                                     imgaug_seq=val_seq,
                                     with_label=False,
                                     test_mode=True, **params)

    df = pd.DataFrame(columns=('image_name', ) + tuple(unique_tags))
    total_counter = 0
    ll = len(test_id_type_list)

    for x, _, info in val_flow:
        y_pred = model.predict(x)
        s = y_pred.shape[0]
        if verbose > 0:
            print("--", total_counter, '/', ll)
        for i in range(s):
            prefix = 'file_' if 'ATest' in info[i][1] else 'test_'
            df.loc[total_counter, :] = (prefix + info[i][0],) + tuple(y_pred[i, :])
            total_counter += 1

    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def create_submission(df, info):
    _df = df.copy()
    thresholds = {
        'agriculture': 0.5,
        'artisinal_mine': 0.5,
        'bare_ground': 0.5,
        'blooming': 0.5,
        'blow_down': 0.5,
        'clear': 0.5,
        'cloudy': 0.5,
        'conventional_mine': 0.5,
        'cultivation': 0.5,
        'habitation': 0.5,
        'haze': 0.5,
        'partly_cloudy': 0.5,
        'primary': 0.5,
        'road': 0.5,
        'selective_logging': 0.5,
        'slash_burn': 0.5,
        'water': 0.5
    }
    for tag in unique_tags:
        _df.loc[:, tag] = _df[tag].apply(lambda x : 1 if x > thresholds[tag] else 0)
    _df['tags'] = _df[unique_tags].apply(lambda x: " ".join(np.array(unique_tags)[np.where(x.values)[0]]), axis=1)
    _df['image_id'] = _df['image_name'].apply(lambda x: int(x[5:]) + (40669 if 'file_' in x else 0))
    _df = _df.sort_values(by='image_id')[['image_name', 'tags']]

    now = datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    sub_file = os.path.join('..', 'results', sub_file)
    _df.to_csv(sub_file, index=False)
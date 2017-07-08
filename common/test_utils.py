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
            df.loc[total_counter, :] = (prefix + info[i][0] + '.jpg',) + tuple(y_pred[i, :])
            total_counter += 1

    df = df.apply(pd.to_numeric, errors='ignore')
    return df


def create_submission(df, info):
    pass
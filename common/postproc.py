

import numpy as np


def pred_threshold(y_pred):
    return (y_pred > 0.5).astype(np.float32)
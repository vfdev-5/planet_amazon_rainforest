import keras.backend as K
from keras.losses import categorical_crossentropy, mae

def jaccard_loss(y_true, y_pred):
    return 1.0 - jaccard_index(y_true, y_pred)
    

def jaccard_index(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_int_index(y_true, y_pred):
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def false_negatives(Y_true, Y_pred):
    return K.sum(K.round(K.clip(Y_true - Y_pred, 0, 1)))


def categorical_crossentropy_with_mae(y_true, y_pred):
    l1 = categorical_crossentropy(y_true, y_pred)
    l2 = mae(y_true, y_pred)
    return l1 + l2


def mae_with_false_negatives(Y_true, Y_pred):
    # 1.0 / 17.0 = 0.058823529411764705
    return mae(Y_true, Y_pred) + 0.058823529411764705 * false_negatives(Y_true, Y_pred)


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

import keras.backend as K
from keras.losses import categorical_crossentropy, mae, binary_crossentropy


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
    return K.mean(K.round(K.clip(Y_true - Y_pred, 0, 1)))


def categorical_crossentropy_with_mae(y_true, y_pred):
    l1 = categorical_crossentropy(y_true, y_pred)
    l2 = mae(y_true, y_pred)
    return l1 + l2


def mae_with_false_negatives(Y_true, Y_pred):
    return mae(Y_true, Y_pred) + false_negatives(Y_true, Y_pred)


def binary_crossentropy_with_false_negatives(Y_true, Y_pred, a=1.0):
    return binary_crossentropy(Y_true, Y_pred) + a * false_negatives(Y_true, Y_pred)


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


# def f2(y_true, y_pred):
#     # from https://www.kaggle.com/teasherm/keras-metric-for-f-score-tf-only
#     y_pred = K.round(K.clip(y_pred, 0, 1))
#     y_correct = K.round(K.clip(y_true * y_pred, 0, 1))
#     sum_true = K.sum(y_true, axis=1)
#     sum_pred = K.sum(y_pred, axis=1)
#     sum_correct = K.sum(y_correct, axis=1)
#     precision = sum_correct / sum_pred
#     recall = sum_correct / sum_true
#     f_score = (5 * precision * recall + K.epsilon()) / (4 * precision + recall + K.epsilon() * K.epsilon())
#     return K.mean(f_score)

import tensorflow as tf


def f2(y_true, y_pred):
    # from https://www.kaggle.com/teasherm/keras-metric-for-f-score-tf-only
    y_true = tf.cast(y_true, "int32")
    y_pred = tf.cast(tf.round(y_pred), "int32") # implicit 0.5 threshold via tf.round
    y_correct = y_true * y_pred
    sum_true = tf.reduce_sum(y_true, axis=1)
    sum_pred = tf.reduce_sum(y_pred, axis=1)
    sum_correct = tf.reduce_sum(y_correct, axis=1)
    precision = sum_correct / sum_pred
    recall = sum_correct / sum_true
    f_score = 5 * precision * recall / (4 * precision + recall)
    f_score = tf.where(tf.is_nan(f_score), tf.zeros_like(f_score), f_score)
    return tf.reduce_mean(f_score)



from sklearn.metrics import fbeta_score


def score(y_true, y_pred):
    average = 'samples' if len(y_true.shape) == 2 else 'binary'
    return fbeta_score(y_true, y_pred, beta=2, average=average)

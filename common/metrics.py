

from sklearn.metrics import fbeta_score


def score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='samples')

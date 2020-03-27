import numpy as np
from ml.preprocessing import *
from ml.utils import pd_to_np_1d


def log_loss(h, y, p=1e-5):
    return -1 / y.shape[0] * np.sum(y * np.log(h + p) + (1 - y) * np.log(1 - h + p))

def category_log_loss(h, y, p=1e-5):
    m, c = y.shape[0], len(np.unique(y))
    y, h = get_one_hot(y, c), get_one_hot(h, c)
    p = p * np.ones((1, c))
    return -1 / m * np.sum(y * np.log(h + p) + (1 - y) * np.log(1 - h + p))

def confusion_matrix(y, h):
    ## this implementation is stricted to the binary classification task
    tp, fp, fn, tn = 0, 0, 0, 0
    m = len(y)
    for i in range(m):
        if y[i] == 1 and y[i] == h[i]:
            tp += 1
        elif y[i] == 1 and y[i] != h[i]:
            fn += 1
        elif y[i] == 0 and y[i] == h[i]:
            tn += 1
        else:
            fp += 1
    return tp, fp, fn, tn

def precision(y, h):
    tp, fp, _, _ = confusion_matrix(y, h)
    return tp / (tp + fp)

def recall(y, h):
    tp, _, fn, _ = confusion_matrix(y, h)
    return tp / (tp + fn)

def TPR(y, h):
    tp, _, fn, _ = confusion_matrix(y, h)
    return tp / (tp + fn)

def FPR(y, h):
    _, fp, _, tn = confusion_matrix(y, h)
    return fp / (tn + fp)

def f1_score(y, h):
    p = precision(y, h)
    r = recall(y, h)
    return 2 * p * r / (p + r)

def weighted_f1_score(y, h, beta=0.9):
    p = precision(y, h)
    r = recall(y, h)
    return (1 + beta ** 2) * p * r / ((beta ** 2) * p + r)

def precision_recall_curve(y, p):
    precisions = []
    recalls = []

    m = len(y)

    y_true = _sort_ytrue_by_proba(y, p)

    y_pred = np.zeros((m,), dtype='int')
    tps, fps, fns, _ = confusion_matrix(y_true, y_pred)
    for i in range(m):
        if y_true[i] == 1:
            tps += 1
            fns -= 1
        else:
            fps += 1

        precisions.append(_precision(tps, fps))
        recalls.append(_recall(tps, fns))

    return precisions, recalls

def roc_curve(y, p):
    tprs, fprs = [], []
    m = len(y)

    y_true = _sort_ytrue_by_proba(y ,p)
    y_pred = np.zeros((m,), dtype='int')
    tps, fps, fns, tns = confusion_matrix(y_true, y_pred)
    for i in range(m):
        if y_true[i] == 1:
            tps += 1
            fns -= 1
        else:
            fps += 1
            tns -= 1

        tprs.append(_recall(tps, fns))
        fprs.append(_fpr(fps, tns))

    return tprs, fprs


def auc(x, y):
    m = len(x)
    auc = 0

    x_prev, y_prev = x[0], y[0]
    for i in range(1, m):
        x_curr, y_curr = x[i], y[i]
        auc += (x_curr - x_prev) * (y_prev + y_curr) / 2
        x_prev, y_prev = x[i], y[i]

    return auc


def _precision(tps, fps):
    return tps / (tps + fps)

def _recall(tps, fns):
    return tps / (tps + fns)

def _fpr(fps, tns):
    return fps / (tns + fps)

def _sort_ytrue_by_proba(y, p):
    arr = []
    for i in range(len(y)):
        arr.append((p[i], y[i]))
    arr.sort()
    arr = arr[::-1]
    arr = np.array(arr)
    y_true = arr[:, 1]

    return y_true

def mean_squared_error(y, h):
    y = pd_to_np_1d(y)
    h = pd_to_np_1d(h)
    return np.average(((y - h) ** 2))

def sum_squared_error(y, h):
    y = pd_to_np_1d(y)
    h = pd_to_np_1d(h)
    return np.sum((y - h) ** 2)

def variance(y):
    y = pd_to_np_1d(y)
    return np.var(y)

def r2_score(y, h):
    u = mean_squared_error(y, h)
    v = variance(y)
    if v == 0:
        return 0.0
    else:
        return 1 - u / v




import numpy as np


def train_test_split(x, y, size=0.8, seed=0):
    """
    Train, Test split to asses scores of models
    :param x: array-like
    :param y: array-like labels
    :param size: float relative dimension of train
    :param seed: random seed
    :return: x_train, y_train, x_test, y_test
    """
    nTrain = int(x.shape[0] * size)
    np.random.seed(seed)
    idx = np.random.permutation(x.shape[0])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    x_train = x[idxTrain, :]
    x_test = x[idxTest, :]
    y_train = y[idxTrain]
    y_test = y[idxTest]
    return x_train, y_train, x_test, y_test


def leave_one_out_split(x, y, seed=0):
    return kfold_split(x, y, len(y), seed)


def kfold_split(x, y, splits=10, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(x.shape[0])
    chunks = np.array_split(idx, splits)
    for i in range(splits):
        x_tr = np.vstack([x[el] for j, el in enumerate(chunks) if j != i])
        y_tr = np.hstack([y[el] for j, el in enumerate(chunks) if j != i])
        x_ts = x[chunks[i]]
        y_ts = y[chunks[i]]
        yield x_tr, y_tr, x_ts, y_ts


def confusion_matrix(l_calc, l_real):
    n_labels = len(np.unique(l_real))
    confusion = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            confusion[i, j] = sum((l_calc == i) & (l_real == j))
    return confusion


def accuracy(l_calc, l_real):
    return sum(l_calc == l_real) / len(l_calc)


def err_rate(l_calc, l_real):
    return 1-accuracy(l_calc, l_real)

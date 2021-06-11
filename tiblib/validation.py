import numpy as np
from matplotlib import pyplot as plt

import tiblib.probability as pr

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


def confusion_matrix(l_calc, l_real, print_cm=False):
    n_labels = len(np.unique(l_real))
    confusion = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            confusion[i, j] = sum((l_calc == i) & (l_real == j))
    if print_cm:
        print("\tFalse negative rate: ", "{:.4f}".format(confusion[1, 0] / (confusion[1, 0] + confusion[1, 1])))
        print("\tFalse positive rate: ", "{:.4f}".format(confusion[0, 1] / (confusion[0, 1] + confusion[0, 0])))
        print("\tTrue positive rate: ", "{:.4f}".format(confusion[1, 1] / (confusion[1, 1] + confusion[1, 0])))
        print("\tTrue negative rate: ", "{:.4f}".format(confusion[0, 0] / (confusion[0, 1] + confusion[0, 0])))
    return confusion


def accuracy(l_calc, l_real):
    return sum(l_calc == l_real) / len(l_calc)


def err_rate(l_calc, l_real):
    return 1-accuracy(l_calc, l_real)


def FScore(l_calc, l_real):
    conf = confusion_matrix(l_calc, l_real)
    return conf[0, 0]/(conf[0, 0]*0.5*(conf[0, 1]+conf[1, 0]))


def grid_search(hypers: dict):
    def rec_grid_search(hyper, i, resul, curr):
        if i == len(hyper):
            resul.append(curr.copy())
            return
        for el in hyper[i][1]:
            curr[hyper[i][0]] = el
            rec_grid_search(hyper, i+1, resul, curr)
            curr[hyper[i][0]] = None

    res = []
    rec_grid_search(list(hypers.items()), 0, res, {})
    return res


def plotROC(llrs, lab, name=None):
    TPR = []
    FPR = []
    index = 0
    #llrs_bet = llrs[np.logical_and(llrs > np.median(llrs) - 5, llrs < np.median(llrs) + 5)]
    llrs_sort = np.sort(llrs)
    for i in llrs_sort:
        pred = np.where(llrs > i+0.000001, 1, 0)
        conf = pr.getConfusionMatrix2(pred, lab)
        TPR.insert(index, conf[1, 1] / (conf[0, 1] + conf[1, 1]))
        FPR.insert(index, conf[1, 0] / (conf[0, 0] + conf[1, 0]))
        index += 1
    plt.plot(np.array(FPR), np.array(TPR), label=name)
    return


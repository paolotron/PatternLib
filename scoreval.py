import re

import numpy as np
from matplotlib import pyplot as plt

from tiblib.probability import minDetectionCost, getConfusionMatrix2
from tiblib.validation import confusion_matrix, plotROC


def score_eval():
    labels = np.load("result/final/labels.npy").astype("int32")
    scores = np.load("result/final/scores.npy")
    pipe = np.load("result/final/pipe.npy")
    '''
    x = np.array([1, 1, 1, 0]).astype("bool")
    y = np.array([1, 0, 1, 0]).astype("bool")
    p1 = confusion_matrix(x, y)
    p2 = getConfusionMatrix2(x, y)
    precision = 3
    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]
        if s.startswith("StandardScaler()->SVM"):
            print(pip)
            mindcf1, _ = minDetectionCost(score, label, 1000, 0.5)
            mindcf2, _ = minDetectionCost(score, label, 1000, 0.1)
            mindcf3, _ = minDetectionCost(score, label, 1000, 0.9)
            print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")
    '''


def plot():
    labels = np.load("result/final/labels.npy").astype("int32")
    scores = np.load("result/final/scores.npy")
    pipe = np.load("result/final/pipe.npy")

    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]
        #plt.plot([0, 0], [1, 1], 'k-')

        if s.startswith("StandardScaler()->LDA(n_feat=6)->LogisticRegression(norm=0.1)"):
            print(pip)
            plotROC(score, label, "Logistic Regression")
        if s.startswith("StandardScaler()->LDA(n_feat=6)->SVM(kernel=Polynomial, C=1, regularization=0, d=2, c=1)"):
            print(pip)
            plotROC(score, label, "SVM Polynomial")
        if s.startswith("StandardScaler()->LDA(n_feat=6)->TiedCovarianceGMM(alpha=0.1, N=4"):
            print(pip)
            plotROC(score, label, "Tied Covariance GMM")
        if s.startswith("StandardScaler()->LDA(n_feat=6)->TiedG"):
            print(pip)
            plotROC(score, label, "Tied Gaussian")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # score_eval()
    plot()


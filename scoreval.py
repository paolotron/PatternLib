import numpy as np
from matplotlib import pyplot as plt

from tiblib.probability import minDetectionCost
from tiblib.probability import normalizedBayesRisk
from tiblib.validation import plotROC

precision = 3


def score_eval():
    labels = np.load("result/final2/labels.npy").astype("int32")
    scores = np.load("result/final2/scores.npy")
    pipe = np.load("result/final2/pipe.npy")
    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]
        if s.startswith("StandardScaler()->SVM"):
            print(pip)
            mindcf1, _ = minDetectionCost(score, label, 1000, 0.5)
            mindcf2, _ = minDetectionCost(score, label, 1000, 0.1)
            mindcf3, _ = minDetectionCost(score, label, 1000, 0.9)
            print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")


def joint_score_eval():
    scores = np.load("result/jointResults/jointscores.npy")
    labels = np.load("result/jointResults/jointlabels.npy")
    for score, label in zip(scores, labels):
        mindcf1, _ = minDetectionCost(score, label, 10000, 0.5)
        mindcf2, _ = minDetectionCost(score, label, 10000, 0.1)
        mindcf3, _ = minDetectionCost(score, label, 10000, 0.9)
        print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")


def plotROCscores():
    labels = np.load("result/final/labels.npy").astype("int32")
    scores = np.load("result/final/scores.npy")
    pipe = np.load("result/final/pipe.npy")

    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]

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
    scores = np.load("result/jointResults/jointscores.npy")
    labels = np.load("result/jointResults/jointlabels.npy")
    plotROC(scores[0], labels[0], "JointModel")
    plt.plot(np.array([[0, 0], [1, 1]]))
    plt.legend()
    plt.savefig("images/ROC.eps", format='eps')
    plt.show()

if __name__ == "__main__":
    score_eval()
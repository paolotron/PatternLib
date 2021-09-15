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
        if s.startswith("StandardScaler()->GMM"):
            print(pip)
            mindcf1, _ = minDetectionCost(score, label, 1000, 0.5)
            mindcf2, _ = minDetectionCost(score, label, 1000, 0.1)
            mindcf3, _ = minDetectionCost(score, label, 1000, 0.9)
            print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")


def joint_score_eval():
    scores = np.load("result/jointResults2/jointscores.npy")
    labels = np.load("result/jointResults2/jointlabels.npy")
    for score, label in zip(scores, labels):
        mindcf1, _ = minDetectionCost(score, label, 10000, 0.5)
        mindcf2, _ = minDetectionCost(score, label, 10000, 0.1)
        mindcf3, _ = minDetectionCost(score, label, 10000, 0.9)
        print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")


def plotROCscores():
    labels = np.load("result/final2/labels.npy").astype("int32")
    scores = np.load("result/final2/scores.npy")
    pipe = np.load("result/final2/pipe.npy")
    labelsSVM = np.load("result/SVMfinal2/labels.npy").astype("int32")
    scoresSVM = np.load("result/SVMfinal2/scores.npy")
    pipeSVM = np.load("result/SVMfinal2/pipe.npy")
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]
        if s.startswith("StandardScaler()->PCA(n_feat=7)->TiedCovarianceGMM(alpha=0.1, N=4"):
            print(pip)
            plotROC(score, label, "Tied Covariance GMM")
        if s.startswith("StandardScaler()->PCA(n_feat=7)->TiedG"):
            print(pip)
            plotROC(score, label, "Tied Gaussian")
        if s.startswith("StandardScaler()->PCA(n_feat=7)->SVM(kernel=Linear, C=0.5, regularization=5)"):
            print(pip)
            plotROC(score, label, "SVM Linear")

    for pip, score, label in zip(pipeSVM, scoresSVM, labelsSVM):
        s: str = pip[0]
        if s.startswith("StandardScaler()->PCA(n_feat=7)->SVM(kernel=Radial, C=10, regularization=1, gamma=0.05)"):
            print(pip)
            plotROC(score, label, "SVM Radial")

    # scores = np.load("result/jointResults2/jointscores.npy")
    # labels = np.load("result/jointResults2/jointlabels.npy")
    # plotROC(scores[0], labels[0], "JointModel")
    plt.plot(np.array([[0, 0], [1, 1]]))
    plt.legend()
    plt.savefig("images/ROC.eps", format='eps')
    plt.show()

if __name__ == "__main__":
    score_eval()
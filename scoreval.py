import re

import numpy as np

from tiblib.probability import minDetectionCost, getConfusionMatrix2
from tiblib.validation import confusion_matrix


def score_eval():
    labels = np.load("result/final/labels.npy").astype("int32")
    scores = np.load("result/final/scores.npy")
    pipe = np.load("result/final/pipe.npy")

    x = np.array([1, 1, 1, 0]).astype("bool")
    y = np.array([1, 0, 1, 0]).astype("bool")
    p1 = confusion_matrix(x, y)
    p2 = getConfusionMatrix2(x, y)
    precision = 3
    for pip, score, label in zip(pipe, scores, labels):
        s: str = pip[0]
        if s.startswith("StandardScaler()->LDA(n_feat=6)"):
            print(pip)
            mindcf1, _ = minDetectionCost(score, label, 1000, 0.5)
            mindcf2, _ = minDetectionCost(score, label, 1000, 0.1)
            mindcf3, _ = minDetectionCost(score, label, 1000, 0.9)
            print(f"& {round(mindcf1, precision)} & {round(mindcf2, precision)} & {round(mindcf3, precision)} \\\\")


if __name__ == "__main__":
    score_eval()

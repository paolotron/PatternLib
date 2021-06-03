import numpy as np
import matplotlib.pyplot as plt

import tiblib
import tiblib.validation as val
import tiblib.Classifier as cl
from tiblib.blueprint import Faucet
from tiblib.preproc import StandardScaler
import sklearn.datasets

attributes = [
    "Mean of the integrated profile",
    "Standard deviation of the integrated profile",
    "Excess kurtosis of the integrated profile",
    "Skewness of the integrated profile",
    "Mean of the DM-SNR curve",
    "Standard deviation of the DM-SNR curve",
    "Excess kurtosis of the DM-SNR curve",
    "Skewness of the DM-SNR curve"
]


def get_pulsar_data(path_train="Data/Train.txt", path_test="Data/Test.txt", labels=False):
    train_data = np.loadtxt(path_train, delimiter=",")
    test_data = np.loadtxt(path_test, delimiter=",")
    if labels:
        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    else:
        return train_data, test_data


def test_random_models():
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    model: Faucet
    whitener = StandardScaler()
    whitener.fit(train, None)
    for model in (cl.LogisticRegression(norm_coeff=0.1), cl.GaussianClassifier(), cl.TiedGaussian(), cl.NaiveBayes()):
        model.fit(whitener.transform(train), train_labels)
        prediction = model.predict(whitener.transform(test))
        conf_matrix = val.err_rate(prediction, test_labels)
        print(f"Model: {type(model).__name__}, err_rate: {conf_matrix}")


def plot_data_exploration():
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    fig, axes = plt.subplots(train.shape[1], 1, figsize=(10, 15))
    for i in range(train.shape[1]):
        axes[i].hist(train[:, i][train_labels == 0], bins=50, density=True, alpha=0.5)
        axes[i].hist(train[:, i][train_labels == 1], bins=50, density=True, alpha=0.5)
        axes[i].title.set_text(attributes[i])
    fig.tight_layout()
    fig.show()
    fig, axes = plt.subplots(1, train.shape[1], figsize=(10, 10))
    for i in range(train.shape[1]):
        axes[i].boxplot(train[:, i][train_labels == 0])
        axes[i].boxplot(train[:, i][train_labels == 1])
    fig.tight_layout()
    fig.show()


def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'], sklearn.datasets.load_iris()['target']
    D = D[L != 0, :]
    L = L[L != 0]
    L[L == 2] = 0
    return D, L


def test_svm():
    data, label = load_iris_binary()
    DTR, LTR, DTE, LTE = val.train_test_split(data, label, 2/3)
    for hyper in [(1, 0.1), (1, 1.), (1, 10.), (10, 0.1), (10, 1.), (10, 10.)]:
        model = tiblib.Classifier.SVM(*hyper)
        model.fit(DTR, LTR)
        prediction = model.predict(DTE)
        print(
            f"K: {hyper[0]} C: {hyper[1]} PrimalLoss: {model.primal} DualityGap: {model.get_gap()} error rate: {val.err_rate(prediction, LTE)}")

    for hyper in [(0, 1, "Poly", (2, 0)), (1, 1, "Poly", (2, 0)), (0, 1, "Poly", (2, 1)), (1, 1, "Poly", (2, 1)),
                  (0, 1, "Radial", (1,)), (0, 1, "Radial", (10,)), (1, 1, "Radial", (1,)), (1, 1, "Radial", (10,))]:
        model = tiblib.Classifier.SVM(*hyper)
        model.fit(DTR, LTR)
        prediction = model.predict(DTE)
        print(f"K: {hyper[0]}, C: {hyper[1]}, Ker: {hyper[2]}, {hyper[3]}, err: {val.err_rate(prediction, LTE)}")


if __name__ == "__main__":
    test_svm()



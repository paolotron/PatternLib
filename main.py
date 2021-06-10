import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

import tiblib.Classifier
import tiblib.pipeline as pip
import tiblib.probability as prob
import tiblib.validation as val
import tiblib.Classifier as cl
import tiblib.preproc as prep
from tiblib.blueprint import Faucet
from tiblib.preproc import StandardScaler
from datetime import datetime

save_logs = True
all_scores = []
all_pipes = []
all_labels = []

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
    for model in (cl.LogisticRegression(norm_coeff=0.1), cl.GaussianClassifier(), cl.TiedGaussian(), cl.NaiveBayes(),
                  cl.GaussianMixture(alpha=0.1), cl.SVM(ker="Poly", paramker=[2, 1])):
        model.fit(whitener.transform(train), train_labels)
        prediction = model.predict(whitener.transform(test))
        err_rate = val.err_rate(prediction, test_labels)
        print(f"Model: {type(model).__name__}\n\tError rate:", "{:.4f}".format(err_rate))
        _ = val.confusion_matrix(prediction, test_labels, print_cm=True)


def kfold_test():
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    pipe_list: List[pip.Pipeline]
    preprocessing_pipe_list = [
        pip.Pipeline([prep.StandardScaler()]),
        # pip.Pipeline([prep.StandardScaler(), prep.Pca(train.shape[1])]),
        # pip.Pipeline([prep.StandardScaler(), prep.Pca(train.shape[1] - 1)]),
        # pip.Pipeline([prep.StandardScaler(), prep.Lda(train.shape[1])]),
        # pip.Pipeline([prep.StandardScaler(), prep.Lda(train.shape[1] - 1)]),
        pip.Pipeline([prep.StandardScaler(), prep.Lda(train.shape[1] - 2)]),
        # pip.Pipeline([prep.StandardScaler(), prep.Pca(train.shape[1]), prep.Lda(train.shape[1])]),
        # pip.Pipeline([prep.StandardScaler(), prep.Pca(train.shape[1] - 1), prep.Lda(train.shape[1] - 1)]),
    ]
    K = 5

    # Gaussian
    for pipe in preprocessing_pipe_list:
        test_model(pipe, cl.GaussianClassifier, None, train, train_labels)
    # Naive Bayes
    for pipe in preprocessing_pipe_list:
        test_model(pipe, cl.NaiveBayes, None, train, train_labels)
    # Tied Gaussian
    for pipe in preprocessing_pipe_list:
        test_model(pipe, cl.TiedGaussian, None, train, train_labels)
    #   Logistic regression
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'norm_coeff': [0.1, 0.5, 1]}):
            test_model(pipe, cl.LogisticRegression, hyper, train, train_labels)
    #   SVM no kern
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [0.1, 1, 10], 'k': [1, 10], 'ker': [None]}):
            test_model(pipe, cl.SVM, hyper, train, train_labels)
    #   SVM Poly
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [1], 'k': [0, 1], 'ker': ['Poly'], 'paramker': [[2, 0], [2, 1]]}):
            test_model(pipe, cl.SVM, hyper, train, train_labels)
    #   SVM Radial
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [1], 'k': [0, 1], 'ker': ['Radial'], 'paramker': [[1], [10]]}):
            test_model(pipe, cl.SVM, hyper, train, train_labels)
    #   Gaussian Mixture tied
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'tied': [True], 'alpha': [0.1], 'N': [0, 1, 2]}):
            test_model(pipe, cl.GaussianMixture, hyper, train, train_labels)
    #   Gaussian Mixture diag
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'diag': [False], 'alpha': [0.1], 'N': [0, 1, 2]}):
            test_model(pipe, cl.GaussianMixture, hyper, train, train_labels)
    #   Gaussian Mixture psi
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'psi': [0.01], 'alpha': [0.1], 'N': [0, 1, 2]}):
            test_model(pipe, cl.GaussianMixture, hyper, train, train_labels)


def test_model(pipe: pip.Pipeline, mod, hyper: dict, train: np.ndarray, train_labels: np.ndarray):
    model = mod(**hyper if hyper else {})
    pipe.add_step(model)
    err_rate = k_test(pipe, train, train_labels, 5)
    print(f"{pipe}:{err_rate}")
    pipe.rem_step()


def k_test(pipe, train, train_labels, K):
    err_rate = 0
    scores = np.empty((0,), float)
    labels = np.empty((0,), int)
    for x_tr, y_tr, x_ev, y_ev in val.kfold_split(train, train_labels, K):
        pipe.fit(x_tr, y_tr)
        lab, ratio = pipe.predict(x_ev, True)
        err_rate += val.err_rate(lab, y_ev) / K
        scores = np.append(scores, ratio, axis=0)
        labels = np.append(labels, y_ev, axis=0)
    save_scores(scores, pipe, labels)
    return  err_rate


def save_scores(score, pipe, label):
    all_scores.append(score)
    all_pipes.append(pipe.__str__())
    all_labels.append(label)
    np.save(f"{logfile}/scores", np.vstack(all_scores))
    np.save(f"{logfile}/labels", np.vstack(all_labels))
    np.save(f"{logfile}/pipe", np.vstack(all_pipes))


logfile = None

if save_logs:
    logfile = datetime.now().strftime("result/%d-%m-%Y-%H-%M-%S")

if __name__ == "__main__":
    if save_logs:
        os.mkdir(logfile)
    kfold_test()
    print("Finito")



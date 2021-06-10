import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

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
        model = cl.GaussianClassifier()
        pipe.add_step(model)
        dcf, err_rate = k_test(pipe, train, train_labels, 5)
        pipe.rem_step()
        print("Gaussian", " error rate: ", err_rate, " ", pipe)
        print("\tmean min_DCF: ", dcf)

    # Naive Bayes
    for pipe in preprocessing_pipe_list:
        model = cl.NaiveBayes()
        pipe.add_step(model)
        dcf, err_rate = k_test(pipe, train, train_labels, 5)
        pipe.rem_step()
        print("Naive Bayes", " error rate: ", err_rate, " ", pipe)
        print("\tmean min_DCF: ", dcf)
    # Tied Gaussian
    for pipe in preprocessing_pipe_list:
        model = cl.TiedGaussian()
        pipe.add_step(model)
        dcf, err_rate = k_test(pipe, train, train_labels, 5)
        print("Tied Gaussian", " error rate: ", err_rate, " ", pipe)
        print("\tmean min_DCF: ", dcf / K)
        pipe.rem_step()
    #   Logistic regression
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'norm_coeff': [0.1, 0.5, 1]}):
            model = cl.LogisticRegression(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("Logistic regression", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)
    #   SVM no kern
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [0.1, 1, 10], 'k': [1, 10], 'ker': [None]}):
            model = cl.SVM(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("SVM(no ker)", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)
    #   SVM Poly
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [1], 'k': [0, 1], 'ker': ['Poly'], 'paramker': [[2, 0], [2, 1]]}):
            model = cl.SVM(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("SVM Poly", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)
    #   SVM Radial
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'c': [1], 'k': [0, 1], 'ker': ['Radial'], 'paramker': [[1], [10]]}):
            model = cl.SVM(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("SVM Radial", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)
    #   Gaussian Mixture tied
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'tied': [True], 'alpha': [0.1], 'N': [0, 1, 2]}):
            model = cl.GaussianMixture(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("Gaussian Mixture Tied", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)
    #   Gaussian Mixture diag
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'diag': [False], 'alpha': [0.1], 'N': [0, 1, 2]}):
            model = cl.GaussianMixture(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("Gaussian Mixture Diag", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)
    #   Gaussian Mixture psi
    for pipe in preprocessing_pipe_list:
        for hyper in val.grid_search({'psi': [0.01], 'alpha': [0.1], 'N': [0, 1, 2]}):
            model = cl.GaussianMixture(**hyper)
            pipe.add_step(model)
            dcf, err_rate = k_test(pipe, train, train_labels, 5)
            pipe.rem_step()
            print("Gaussian Mixture Psi", " error rate: ", err_rate, " ", hyper, " ", pipe)
            print("\tmean min_DCF: ", dcf)


def k_test(pipe, train, train_labels, K, prior_prob=0.091):
    err_rate = 0
    dcf = 0
    scores = np.empty((0,), float)
    labels = np.empty((0,), int)
    for x_tr, y_tr, x_ev, y_ev in val.kfold_split(train, train_labels, K):
        pipe.fit(x_tr, y_tr)
        lab, ratio = pipe.predict(x_ev, True)
        err_rate += val.err_rate(lab, y_ev) / K
        scores = np.append(scores, ratio, axis=0)
        labels = np.append(labels, y_ev, axis=0)
    # dcf = prob.minDetectionCost(scores, labels.astype(int), pi1=prior_prob)
    save_scores(scores, pipe, labels)
    return dcf, err_rate


def save_scores(score, pipe, label):
    all_scores.append(score)
    all_pipes.append(pipe.__str__())
    all_labels.append(label)
    np.save(f"{logfile}/scores", np.vstack(all_scores))
    np.save(f"{logfile}/labels", np.vstack(all_labels))
    np.save(f"{logfile}/pipe", np.vstack(all_pipes))


def save_res_to_file(pipe: pip.Pipeline, minDCF: float, err_rate: float):
    if save_logs:
        file = open(logfile, 'a')
        file.write(f"{pipe}: MinDCF={minDCF} err_rate={err_rate}\n")
        file.close()


if save_logs:
    logfile = datetime.now().strftime("result/%d-%m-%Y-%H-%M-%S")

if __name__ == "__main__":
    if save_logs:
        os.mkdir(logfile)
    kfold_test()
    print("Finito")



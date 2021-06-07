from typing import List

import numpy as np
import matplotlib.pyplot as plt

import tiblib.pipeline as pip
import tiblib.validation as val
import tiblib.Classifier as cl
import tiblib.preproc as prep
from tiblib.blueprint import Faucet
from tiblib.preproc import StandardScaler

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
    pipe = pip.Pipeline([prep.Pca(train.shape[1]-1), cl.GaussianClassifier()])
    pipe.fit(train, train_labels)
    print(val.err_rate(pipe.predict(test), test_labels))
    pipe_list: List[pip.Pipeline]
    preprocessing_pipe_list = [
        pip.Pipeline([]),
        pip.Pipeline([prep.Pca(train.shape[1])]),
        pip.Pipeline([prep.Pca(train.shape[1] - 1)]),
        pip.Pipeline([prep.Lda(train.shape[1])]),
        pip.Pipeline([prep.Lda(train.shape[1]) - 1]),
        pip.Pipeline([prep.Pca(train.shape[1]), prep.Lda(train.shape[1])]),
        pip.Pipeline([prep.Pca(train.shape[1] - 1), prep.Lda(train.shape[1])]),
        pip.Pipeline([prep.Pca(train.shape[1] - 1), prep.Lda(train.shape[1] - 1)])
    ]
    for pipe in preprocessing_pipe_list:
        for x_tr, y_tr, x_ev, y_ev in val.kfold_split(train, train_labels, 5):
            pipe.fit(x_tr, y_tr)

            score = pipe.predict(x_ev, True)
            # TODO EVALUATE SCORE



if __name__ == "__main__":
    kfold_test()



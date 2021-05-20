import numpy as np
import matplotlib as plt
import tiblib.validation as val
import tiblib.Classifier as cl
from tiblib.blueprint import Faucet
from tiblib.preproc import StandardScaler


def get_pulsar_data(path_train="Data/Train.txt", path_test="Data/Test.txt", labels=False):
    train_data = np.loadtxt(path_train, delimiter=",")
    test_data = np.loadtxt(path_test, delimiter=",")
    if labels:
        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    else:
        return train_data, test_data


def main():
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    model: Faucet
    whitener = StandardScaler()
    whitener.fit(train, None)
    for model in (cl.LogisticRegression(norm_coeff=0.1), cl.GaussianClassifier(), cl.TiedGaussian(), cl.NaiveBayes()):
        model.fit(whitener.transform(train), train_labels)
        prediction = model.predict(whitener.transform(test))
        conf_matrix = val.err_rate(prediction, test_labels)
        print(f"Model: {type(model).__name__}, err_rate: {conf_matrix}")


if __name__ == "__main__":
    main()

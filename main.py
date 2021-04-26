import numpy as np
import matplotlib as plt


def get_pulsar_data(path_train="Data/Train.txt", path_test="Data/Test.txt", labels=False):
    train_data = np.loadtxt(path_train, delimiter=",")
    test_data = np.loadtxt(path_test, delimiter=",")
    if labels:
        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]
    else:
        return train_data, test_data


if __name__ == "__main__":
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)

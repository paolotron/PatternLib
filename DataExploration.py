import numpy as np
from matplotlib import pyplot as plt
import tiblib.pipeline as pip
from typing import List
import tiblib.preproc
from modeleval import get_pulsar_data, attributes


def plot_data_exploration(save=False):
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    fig, axes = plt.subplots(train.shape[1], 1, figsize=(10, 15))
    for i in range(train.shape[1]):
        axes[i].hist(train[:, i][train_labels == 0], bins=50, density=True, alpha=0.5, color="blue")
        axes[i].hist(train[:, i][train_labels == 1], bins=50, density=True, alpha=0.5, color="orange")
        axes[i].title.set_text(attributes[i])
    fig.tight_layout()
    if not save:
        fig.show()
    else:
        fig.savefig('images/distribution.eps', format='eps')


def print_stats(latek=False):
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    for i, att in enumerate(attributes):
        x = train[:, i]
        if latek:
            print(f"{att} & {round(min(x), 2)} & {round(max(x), 2)} & {round(np.mean(x), 2)} & {round(np.var(x), 2)}",
                  "\\\\")
        else:
            print(f"{att}: min: {min(x)} max: {max(x)} mean: {np.mean(x)}")


def print_label_stats():
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    for i, att in enumerate(np.unique(train_labels)):
        y = att == train_labels
        print(att, ":", sum(y), ":", sum(y) / len(y))


def plot_covariance(save=False):
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    cov = np.corrcoef(train.T)
    fig, ax = plt.subplots()
    ax.matshow(cov, cmap='bwr')
    for (i, j), z in np.ndenumerate(cov):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    if not save:
        fig.show()
    else:
        fig.savefig('images/covariance.eps', format='eps')


def plot_outliers(save=False):
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    fig, ax = plt.subplots(train.shape[1], 1, figsize=(10, 15))
    train = tiblib.preproc.StandardScaler().fit_transform(train)
    z = tiblib.preproc.get_z_score(train)
    for i in range(train.shape[1]):
        ax[i].hist(z[:, i], bins=100)
        ax[i].title.set_text(attributes[i])
    fig.tight_layout()
    print(sum(z>10))
    if not save:
        fig.show()
    else:
        fig.savefig('images/zscore.eps', format='eps')

def plot_LDA(raw=True, lda=True):
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    if raw:
        fig, ax = plt.subplots(8,8,figsize=(50,50))
        fig.suptitle('Raw Data')
        for i in range(train.shape[1]):
            for j in range(train.shape[1]):
                # plt.figure()
                # plt.grid()
                # title = "Raw data: attributes " + str(i) + " and  " + str(j)
                # plt.title(title)
                # plt.plot(train[train_labels==0, i], train[train_labels==0, j], 'o')
                # plt.plot(train[train_labels == 1, i], train[train_labels == 1, j], 'o')
                # plt.show()
                title = "Attributes " + str(i) + " and  " + str(j)
                ax[i, j].set_title(title)
                ax[i, j].plot(train[train_labels==0, i], train[train_labels==0, j], 'o')
                ax[i, j].plot(train[train_labels==1, i], train[train_labels==1, j], 'o')
        fig.show()

    if lda:
        train_LDA = tiblib.preproc.Lda(n_dim=train.shape[1] - 2).fit_transform(train, train_labels)

        fig2, ay = plt.subplots(train_LDA.shape[1], train_LDA.shape[1], figsize=(50, 50))
        fig2.suptitle('Lda Data')
        for i in range(train_LDA.shape[1]):
            for j in range(train_LDA.shape[1]):
                # if i != j and i > j:
                    # plt.figure()
                    # plt.grid()
                    # title = "LDA data: attributes " + str(i) + " and  " + str(j)
                    # plt.title(title)
                    # plt.plot(train_LDA[train_labels==0, i], train_LDA[train_labels==0, j], 'o')
                    # plt.plot(train_LDA[train_labels==1, i], train_LDA[train_labels==1, j], 'o')
                    # plt.show()
                title = "Attributes " + str(i) + " and  " + str(j)
                ay[i, j].set_title(title)
                ay[i, j].plot(train_LDA[train_labels == 0, i], train_LDA[train_labels == 0, j], 'o')
                ay[i, j].plot(train_LDA[train_labels == 1, i], train_LDA[train_labels == 1, j], 'o')
        fig2.show()
if __name__ == "__main__":
    # print_stats(latek=True)
    # print_label_stats()
    # plot_data_exploration()
    # plot_covariance()
    # plot_outliers(save=True)
    plot_LDA(False, True)

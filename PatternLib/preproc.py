from tiblib.blueprint import Pipe
from tiblib.blueprint import NoFitError
import numpy.linalg as ln
from scipy.linalg import eigh
import numpy as np
from itertools import combinations_with_replacement
from itertools import combinations


class Pca(Pipe):

    def __init__(self, n_dim=None, center=False):
        """
        Pca init method
        :param n_dim: int number of dimensions of the output data
        :param center: Bool default False, if True the output is centered
        """
        self.X = None
        self.P = None
        self.n_dim = n_dim
        self.mu = None
        self.center = center

    def fit(self, x, y=None):
        """
        calculate the P matrix through eig-decomposition
        of the Covariance Matrix
        :param x: array-like
        :param y: None, just for compatibility
        :return: None
        """
        if self.n_dim is None:
            self.n_dim = x.shape[1]
        self.X = x
        mu = self.X.mean(axis=0)
        self.mu = mu
        DC = self.X - mu
        COV = (DC.T @ DC) / self.X.shape[0]
        s, U = ln.eigh(COV)
        self.P = U[:, ::-1][:, 0:self.n_dim]

    def fit_transform(self, x, y=None):
        """
        Compute fit and transform on the same input data
        :param x: array-like
        :param y: None
        :return: array-like
        """
        self.fit(x, None)
        return self.transform(x)

    def transform(self, x):
        """
        Transform the data with the P computed already
        :param x: array-like
        :return: array-like
        """
        if x is None:
            raise NoFitError()
        return (self.P.T @ (x - (self.mu if self.center else 0)).T).T

    def __str__(self):
        return f"PCA(n_feat={self.n_dim})"


class Lda(Pipe):

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def __init__(self, n_dim=None, center=False):
        """
        Lda init method
        :param n_dim: int number of dimensions of the output data
        :param center: Bool default False, if True the output is centered
        """
        self.X = None
        self.Y = None
        self.m = n_dim
        self.U = None
        self.mu = None
        self.center = center

    def transform(self, x):
        """
        Compute the output through the U matrix on the input data
        :param x:
        :return: array-like
        """
        if self.X is None:
            raise NoFitError()
        return (self.U.T @ (x - (self.mu.T if self.center else 0)).T).T

    def fit(self, x, y):
        """
        Compute the U matrix with the between and in-between classes covariance matrixes
        :param x: array-like
        :param y: array-like, classes of x
        :return: None
        """
        if self.m is None:
            self.m = x.shape[0]
        self.X = x.T
        self.Y = y

        mu = self.X.mean(1).reshape((-1, 1))
        self.mu = mu
        SB_ls = []
        SW_ls = []
        for label in set(y):
            x_c = self.X[:, y == label]
            mu_c = x_c.mean(1).reshape((-1, 1))
            nc = x_c.shape[1]
            SB_ls.append(nc * (mu_c - mu) @ (mu_c - mu).T)
            SW_ls.append((x_c - mu_c) @ (x_c - mu_c).T)

        N = self.X.shape[1]
        SB = sum(SB_ls) / N  # Between Class Variability Matrix
        SW = sum(SW_ls) / N  # Within Class Variability Matrix

        s, U = eigh(SB, SW)
        self.U = U[:, ::-1][:, :self.m]

    def __str__(self):
        return f"LDA(n_feat={self.m})"


class StandardScaler(Pipe):

    def __init__(self, with_mean=True, with_std=True):
        self._with_mean = with_mean
        self._with_std = with_std
        self.mu = None
        self.std = None

    def fit(self, x, y=None):
        self.mu = x.mean(0)
        self.std = x.std(0)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def transform(self, x):
        if self.mu is None:
            raise NoFitError()
        mu = self.mu if self._with_mean else 0
        std = self.std if self._with_std else 1
        return (x - mu) / std

    def __str__(self):
        return "StandardScaler()"


class PolynomialFeatures(Pipe):

    def __init__(self, degree=2, interact_only=False):
        self._degree = degree
        self._interact_only = interact_only
        self.n_input_features = 0

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def fit(self, x, y=None):
        self.n_input_features = x.shape[1]

    def transform(self, x):
        if self.n_input_features is None:
            raise NoFitError()
        if self.n_input_features != x.shape[1]:
            raise ValueError("Fitted data has different shape")

        out = np.hstack([np.ones((x.shape[0], 1)), x])
        ls = []
        iterator = combinations_with_replacement(out.T, self._degree) \
            if self._interact_only else combinations(out.T, self._degree)
        for t in iterator:
            ls.append(np.prod(np.array(t), axis=0).reshape(-1, 1))
        return np.hstack(ls)


def get_cov(x: np.ndarray, rt_mean=False):
    M = x.shape[0]
    mu = x.mean(axis=0)
    if rt_mean:
        return (x - mu).T @ (x - mu) / M, mu
    else:
        return (x - mu).T @ (x - mu) / M


def get_z_score(x: np.ndarray):
    return (x-np.mean(x))/np.std(x)


import scipy.special
from .blueprint import *
import numpy as np
from .probability import GAU_ND_logpdf
from .preproc import get_cov
import tiblib
from scipy.optimize import fmin_l_bfgs_b


class Perceptron(Faucet):

    def __init__(self, iterations=100, alpha=0.1, seed=None):
        self.Y = None
        self.X = None
        self.labels = None
        self.iter = iterations
        self.alpha = alpha
        self.weights = None
        self.seed = seed

    def fit(self, x, y):
        if self.seed is not None:
            np.random.seed(self.seed)

        rand = np.random.choice(x.shape[0], x.shape[0])
        self.X = np.hstack([np.ones((x.shape[0], 1)), x])[rand, :].T
        self.Y = y.reshape((-1, 1))[rand, :]
        self.labels = np.unique(y)

        def single(y_true):
            w = np.random.random((self.X.shape[0], 1)).T - 0.5
            for i in range(self.iter):
                for data, y_tr in zip(self.X.T, y_true.T):
                    data = data.reshape((1, -1))
                    y_predict = np.heaviside(w @ data.T, 0)
                    w = w + self.alpha * (y_tr - y_predict) * data
            return w

        lis = []
        for label in self.labels:
            y_l = (self.Y == label)
            lis.append(single(y_l.T))
        self.weights = np.vstack(lis).T
        return self

    def predict(self, x):
        one = np.ones((x.shape[0], 1))
        x = np.hstack([one, x]).T
        y = self.weights.T @ x
        return self.labels[np.argmax(y, axis=0)]

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class GaussianClassifier(Faucet):

    def __init__(self):
        self.estimates = []
        self.labels = None
        self.Spost = None

    def fit(self, x, y):
        un, con = np.unique(y, return_counts=True)
        for label, count in zip(un, con):
            matrix = x[y == label, :]
            self.estimates.append((label, np.mean(matrix, 0), tiblib.preproc.get_cov(matrix), count / x.shape[0]))
        return self

    def predict(self, x, get_prob=False):
        scores = []
        for label, mu, cov, prob in self.estimates:
            scores.append(tiblib.probability.GAU_ND_logpdf(x.T, mu.reshape(-1, 1), cov) + np.log(prob))
        SJoint = np.hstack([value.reshape(-1, 1) for value in scores])
        logsum = scipy.special.logsumexp(SJoint, axis=1)
        self.Spost = SJoint - logsum.reshape(1, -1).T
        res = np.argmax(self.Spost, axis=1)
        return res if not get_prob else np.exp(self.Spost)

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class NaiveBayes(GaussianClassifier):

    def fit(self, x, y):
        un, con = np.unique(y, return_counts=True)
        for label, count in zip(un, con):
            matrix = x[y == label, :]
            cov = np.diag(np.var(matrix, 0))
            self.estimates.append((label, np.mean(matrix, 0), cov, count / x.shape[0]))
        return self


class TiedGaussian(GaussianClassifier):

    def fit(self, x, y):
        super().fit(x, y)
        sigma = (1 / y.shape[0]) * sum([sigma * sum(y == label) for label, mu, sigma, prob in self.estimates])
        self.estimates = [(label, mu, sigma, prob) for label, mu, _, prob in self.estimates]
        return self


class MultiNomial(Faucet):

    def __init__(self, eps=0.01):
        self.eps = eps
        self.labels = None
        self.estimate = None
        self.Spost = None

    def fit(self, x, y):
        self.labels = np.unique(y)
        arr_list = []
        for label in self.labels:
            arr = x[y == label, :].sum(axis=0) + self.eps
            arr = np.log(arr) - np.log(arr.sum())
            arr_list.append(arr)
        self.estimate = np.vstack(arr_list)

    def predict(self, x, get_prob=False):
        sjoint = x @ self.estimate.T
        logsum = scipy.special.logsumexp(sjoint, axis=1)
        self.Spost = sjoint - logsum.reshape(-1, 1)
        res = self.labels[np.argmax(x @ self.estimate.T, axis=1)]
        return res if not get_prob else np.exp(self.Spost)

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)


class LogisticRegression(Faucet):

    def __init__(self, norm_coeff=1, max_fun=15000, max_iter=15000):
        self.norm_coeff = norm_coeff
        self.max_fun = max_fun
        self.max_iter = max_iter
        self.w = None
        self.b = None
        self.binary = None

    def fit(self, x, y):
        if np.unique(y).shape[0] == 2:
            self.fit_2class(x, y)
            self.binary = True
        else:
            self.fit_nclass(x, y)
            self.binary = False

    def fit_2class(self, x, y):
        def objective_function(arr):
            w, b = arr[:-1].reshape(-1, 1), arr[-1]
            regular = self.norm_coeff / 2 * np.power(np.linalg.norm(w.T, 2), 2)
            s = w.T @ x.T + b
            body = y * np.log1p(np.exp(-s)) + (1 - y) * np.log1p(np.exp(s))
            return regular + np.sum(body) / y.shape[0]

        m, f, d = fmin_l_bfgs_b(objective_function, np.zeros(x.shape[1] + 1),
                                approx_grad=True,
                                maxfun=self.max_fun,
                                maxiter=self.max_iter)
        self.w = m[:-1]
        self.b = m[-1]

    def fit_nclass(self, x, y):
        T = np.array((y.reshape(-1, 1) == np.unique(y)), dtype="int32")
        sh = (np.unique(y).shape[0], x.shape[1] + 1)

        def objective_function(arr):
            arr = arr.reshape(sh)
            w, b = arr[:, :-1], arr[:, -1].reshape(-1, 1)
            regular = self.norm_coeff / 2 * np.sum(w * w)
            s = w @ x.T
            ls = scipy.special.logsumexp(s + b)
            ly = s + b - ls
            return regular - np.sum(T * ly.T) / x.shape[0]

        init_w = np.zeros(sh)
        m, f, d = fmin_l_bfgs_b(objective_function, init_w,
                                approx_grad=True,
                                maxfun=self.max_fun,
                                maxiter=self.max_iter)
        m = m.reshape(sh)
        self.w = m[:, :-1]
        self.b = m[:, -1].reshape(-1, 1)

    def predict(self, x, score=False):
        if self.w is None:
            raise NoFitError()
        if self.binary:
            sc = self.w @ x.T + self.b
            return sc if score else sc > 0
        else:
            sc = self.w @ x.T + self.b
            return sc if score else np.argmax(sc, axis=0)

    def fit_predict(self, x, y):
        return self.predict(self.fit(x, y))


class SVM(Faucet):

    def __init__(self, k=1, c=1, ker=None, paramker=None):
        self.C = c
        self.K = k
        self.W = None
        self.b = None
        self.w_cap = None
        self.dual_gap = None
        self.primal = None
        self.dual = None
        self.ker = ker
        self.paramker = paramker
        self.z = None
        self.x = None
        self.alpha = None

    def fit(self, x, y):
        z = np.where(y, -1, 1).reshape(-1, 1)
        D = np.vstack([x.T, self.K * np.ones((1, x.shape[0]))])
        H = z.T * (z * (self.kernel(x, x)))
        self.z = z
        self.x = x

        def obj_fun(vect):
            alpha = vect.reshape(-1, 1)
            L = 0.5 * alpha.T @ H @ alpha - np.sum(alpha)
            grad = (H @ alpha - 1).reshape(-1, )
            L = L.reshape(-1, )
            return L, grad

        init_alpha = np.ones((H.shape[0],))
        bounds = [(0, self.C) for _ in range(H.shape[0])]
        m, self.primal, _ = fmin_l_bfgs_b(obj_fun, init_alpha, bounds=bounds, approx_grad=False)
        self.alpha = m
        res = np.sum(m * z.T * D, axis=1)
        self.W = res[:-1]
        self.b = res[-1]
        self.w_cap = res

        const = 0.5 * (res * res).sum()
        decision = self.W @ x.T + self.b * self.K
        tt = np.vstack([1 - z.T * decision, np.zeros(D.shape[1])])
        temp_max = np.max(tt, axis=0)
        primal_obj = const + self.C * np.sum(temp_max)
        primal_obj_func = obj_fun(m)[0]
        self.dual_gap = primal_obj_func + primal_obj

    def kernel(self, xi, xj):
        if self.ker is None:
            return xi @ xj.T + self.K
        if self.ker == "Poly":
            return np.power(xi @ xj.T + self.paramker[1], self.paramker[0]) + self.K**2
        if self.ker == "Radial":
            res = []
            for line in xi:
                res.append(np.exp(-self.paramker[0] * np.linalg.norm(line-xj, 2, axis=1)**2))
            return np.vstack(res)

    def predict(self, x):
        if self.ker is None:
            score = self.W @ x.T + self.b
        else:
            ker_res = self.kernel(self.x, x)
            score = np.sum(self.alpha.reshape(-1, 1) * self.z * ker_res, axis=0)
        return np.where(score > 0, 0, 1)

    def get_gap(self):
        return self.dual_gap[0] * 2

    def fit_predict(self, x, y):
        pass

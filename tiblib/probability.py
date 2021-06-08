import numpy as np
import numpy.linalg as ln
import scipy
import scipy.special

import tiblib.validation as val
from tiblib.preproc import get_cov


def GAU_pdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """
    Probability function of Guassian distribution
    :param x: ndarray input parameters
    :param mu: float mean of the distribution
    :param var: float variance of the distribution
    :return: ndarray probability of each sample
    """
    k = (1 / (np.sqrt(2 * np.pi * var)))
    up = -np.power(x - mu, 2) / (2 * var)
    return k * np.exp(up)


def GAU_logpdf(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    """
    Log probability function of Guassian distribution
    :param x: ndarray input parameters
    :param mu: float mean of the distribution
    :param var: float variance of the distribution
    :return: ndarray log probability of each sample
    """
    return -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var) - np.power(x - mu, 2) / (2 * var)


def GAU_ND_logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Multivarate Gaussian Distribution probability function<
    :param x: ndarray input matrix
    :param mu: ndarray mean vector
    :param cov: ndarray covariance matrix
    :return: ndarray
    """
    M = x.shape[0]
    s, ld = ln.slogdet(cov)
    k = -M * np.log(2 * np.pi) * 0.5 - s * ld * 0.5
    f = x - mu
    res = k - .5 * ((f.T @ ln.inv(cov)).T * f).T.sum(-1)
    return res


def optimal_bayes_decision_with_ratio(llratio, prior_prob, cost_false_neg, cost_false_pos):
    return llratio > -np.log((prior_prob * cost_false_neg) / ((1 - prior_prob) * cost_false_pos))


def bayes_detection_function_with_confusion(confusion_matrix, prior_prob, cost_false_neg, cost_false_pos):
    FNR = confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
    FPR = confusion_matrix[1, 0] / (confusion_matrix[1, 0] + confusion_matrix[0, 0])
    return FNR * cost_false_neg * prior_prob + FPR * (1 - prior_prob) * cost_false_pos


def optimal_bayes_decision_with_threshold(llratio, thresh):
    return llratio > thresh


def minimal_detection_cost(llr, real_labels, prior_prob, cost_false_neg, cost_false_pos, n_samples=100):
    results = []
    normalizer_factor = min(prior_prob * cost_false_neg, (1 - prior_prob) * cost_false_pos)
    for thresh in np.linspace(min(llr), max(llr), n_samples):
        labels = optimal_bayes_decision_with_threshold(llr, thresh)
        confus = val.confusion_matrix(labels, real_labels)
        risk_test = bayes_detection_function_with_confusion(confus, prior_prob, cost_false_neg, cost_false_pos)
        results.append(risk_test / normalizer_factor)
    return min(results)


def logpdf_GMM(X, gmm, nosum=False):
    res = np.zeros((len(gmm), X.shape[1]))
    for i, gm in enumerate(gmm):
        res[i, :] = GAU_ND_logpdf(X, gm[1], gm[2])
        res[i, :] += np.log(gm[0])
    if not nosum:
        return scipy.special.logsumexp(res, axis=0)
    return res


def EM_estimation(X, gmm, deltat=10 ** -6, *, tied=False, psi=None, diag=False, prin=False):
    if psi is not None:
        for i in range(len(gmm)):
            gmm[i] = (gmm[i][0], gmm[i][1], eig_constraint(gmm[i][2], psi))
    while True:
        S = logpdf_GMM(X, gmm, True)
        marg_dens = scipy.special.logsumexp(S, axis=0)
        sigma = np.exp(S - marg_dens)
        Zg = [np.sum(sigma[i, :]) for i in range(sigma.shape[0])]
        Fg = [np.sum(sigma[i, :] * X, axis=1) for i in range(sigma.shape[0])]
        Sg = []
        for i, _ in enumerate(sigma):
            t1 = sigma[i, :] * X
            rs = t1 @ X.T
            Sg.append(rs)
        mug = [(F / Z).reshape(-1, 1) for F, Z in zip(Fg, Zg)]
        covg = [s / z - (m @ m.T) for m, s, z in zip(mug, Sg, Zg)]
        if diag:
            covg = [cov * np.eye(cov.shape[0]) for cov in covg]
        if tied:
            t_cov = sum(covg) / len(covg)
            covg = [t_cov for _ in covg]
        if psi is not None:
            covg = [eig_constraint(cov, psi) for cov in covg]
        wg = [z / sum(Zg) for z in Zg]
        gmm1 = [(i1, i2, i3) for i1, i2, i3 in zip(wg, mug, covg)]
        delta = np.mean(-logpdf_GMM(X, gmm)) + np.mean(logpdf_GMM(X, gmm1))
        if prin:
            print(np.mean(logpdf_GMM(X, gmm1)))
        if delta < deltat:
            return gmm1
        gmm = gmm1


def eig_constraint(cov, psi):
    U, s, _ = np.linalg.svd(cov)
    s[s < psi] = psi
    return U @ (s.reshape(s.size, 1) * U.T)


def LGB_estimation(X, alpha: float, n: int, *, posterior=1., psi=None, tied=False, diag=False, prin=False):
    gmm = [(posterior, np.mean(X, axis=1).reshape(-1, 1), get_cov(X.T))]
    gmm = EM_estimation(X, gmm, psi=psi, tied=tied, prin=prin)
    for i in range(n):
        new_gmm = []
        for gm in gmm:
            U, s, Vh = np.linalg.svd(gm[2])
            d = U[:, 0:1] * s[0] ** 0.5 * alpha
            new_gmm.append((gm[0] / 2, gm[1] + d, gm[2]))
            new_gmm.append((gm[0] / 2, gm[1] - d, gm[2]))
        gmm = EM_estimation(X, new_gmm, psi=psi, tied=tied, diag=diag, prin=prin)
    return gmm


def getConfusionMatrix(predictions, labels, nclass):
    conf = np.zeros((nclass, nclass), dtype=int)
    for i in range(predictions.shape[0]):
        conf[predictions[i], labels[i]] += 1
    return conf


def bayesRisk(conf, Cfn=1, Cfp=1, pi1=0.5):
    fnr = conf[0, 1] / (conf[0, 1] + conf[1, 1])
    fpr = conf[1, 0] / (conf[1, 0] + conf[0, 0])
    return pi1 * Cfn * fnr + (1 - pi1) * Cfp * fpr


def normalizedBayesRisk(conf, Cfn=1, Cfp=1, pi1=0.091):
    B = bayesRisk(conf, Cfn, Cfp, pi1)
    Bdummy = min(pi1 * Cfn, (1 - pi1) * Cfp)
    return B / Bdummy


def minDetectionCost(llrs, lab, n_trys=100, Cfn=1, Cfp=1, pi1=0.091):
    min_dcf = float('inf')
    threshold = 0
    llrs_bet = llrs[np.logical_and(llrs > np.median(llrs)-5, llrs < np.median(llrs)+5)]
    for i in np.linspace(min(llrs_bet), max(llrs_bet), n_trys):
        pred = np.where(llrs > i, 1, 0)
        conf = getConfusionMatrix(pred, lab, 2)
        r = normalizedBayesRisk(conf, Cfn=Cfn, Cfp=Cfp, pi1=pi1)
        if min_dcf > r:
            min_dcf = r
            threshold = i
    return min_dcf

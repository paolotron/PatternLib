import numpy as np
import numpy.linalg as ln
import tiblib.validation as val


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
        results.append(risk_test/normalizer_factor)
    return min(results)

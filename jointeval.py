import numpy as np

import tiblib.Classifier as cl
from modeleval import get_pulsar_data
from tiblib.pipeline import Pipeline, Jointer
from tiblib.preproc import StandardScaler, Lda
from tiblib.probability import minDetectionCost, getConfusionMatrix2, normalizedBayesRisk
from tiblib.validation import grid_search, kfold_split, err_rate


def evaluatejoint():
    """
    Hyper parameter tuning and threshold calibration for the joint model
    """
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    scores_f = []
    labels_f = []
    pip = Pipeline([StandardScaler(),
                    Lda(n_dim=4),
                    Jointer([
                        cl.TiedGaussian(),
                        cl.GaussianMixture(alpha=0.1, N=2, tied=True),
                        cl.LogisticRegression(norm_coeff=0.1),
                        cl.SVM(c=1, k=0, ker='Poly', paramker=[2, 1])
                    ]),
                    StandardScaler()
                    ])
    for hyper in grid_search({'norm_coeff': [.1, .2, .5, 1]}):
        pip.add_step(cl.LogisticRegression(**hyper))
        err_r = 0
        scores = np.empty((0,), float)
        labels = np.empty((0,), int)
        for x_tr, y_tr, x_ev, y_ev in kfold_split(train, train_labels, 5):
            pip.fit(x_tr, y_tr)
            lab, ratio = pip.predict(x_ev, True)
            err_r += err_rate(lab, y_ev)
            scores = np.append(scores, ratio, axis=0)
            labels = np.append(labels, y_ev, axis=0)
        m1 = minDetectionCost(scores, labels, 1000, pi1=0.5)[0]
        m2 = minDetectionCost(scores, labels, 1000, pi1=0.1)[0]
        m3 = minDetectionCost(scores, labels, 1000, pi1=0.9)[0]
        scores_f.append(scores)
        labels_f.append(labels)
        print(pip)
        print(err_r / 5)
        pip.rem_step()
        print(f"{round(m1, 3)} & {round(m2, 3)} & {round(m3, 3)}")
    np.save("result/jointResults2/jointlabels", scores_f)
    np.save("result/jointResults2/jointscores", labels_f)


def test_final():
    """
    Test the chosen model and compute the DCF on the test set, this is the final evaluation for the project
    """
    train, train_labels, test, test_labels = get_pulsar_data(labels=True)
    thresh = -2.4107759457753977
    pip = Pipeline([StandardScaler(),
                    Lda(n_dim=4),
                    Jointer([
                        cl.TiedGaussian(),
                        cl.GaussianMixture(alpha=0.1, N=2, tied=True),
                        cl.LogisticRegression(norm_coeff=0.1),
                        cl.SVM(c=1, k=0, ker='Poly', paramker=[2, 1])
                    ]),
                    StandardScaler(),
                    cl.LogisticRegression(norm_coeff=0.1)
                    ])
    pip.fit(train, train_labels)
    comp_labels = pip.predict(test, return_prob=True)[1]
    pred = np.where(comp_labels > thresh, True, False)
    conf = getConfusionMatrix2(pred, test_labels)
    r = normalizedBayesRisk(conf, Cfn=1, Cfp=1, pi1=.5)
    print(r)
    print(conf)


if __name__ == "__main__":
    test_final()

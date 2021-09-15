import numpy as np

import tiblib.Classifier as cl
from modeleval import get_pulsar_data
from tiblib.pipeline import Pipeline, Jointer
from tiblib.preproc import StandardScaler, Pca
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
                    Jointer([
                        cl.TiedGaussian(),
                        cl.GaussianMixture(alpha=0.1, N=2, tied=True),
                        cl.SVM(c=0.5, k=5),
                        cl.SVM(c=10, k=1, ker='Radial', paramker=[0.05])
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
        m1, t1 = minDetectionCost(scores, labels, 1000, pi1=0.5)
        m2, t2 = minDetectionCost(scores, labels, 1000, pi1=0.1)
        m3, t3 = minDetectionCost(scores, labels, 1000, pi1=0.9)
        scores_f.append(scores)
        labels_f.append(labels)
        print(pip)
        print(t1, t2, t3)
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
    thresh1 = -2.7529140657835325
    thresh2 = -2.1681219948241015
    thresh3 = -3.081358927555268
    pip = Pipeline([StandardScaler(),
                    Jointer([
                        cl.TiedGaussian(),
                        cl.GaussianMixture(alpha=0.1, N=2, tied=True),
                        cl.SVM(c=0.5, k=5),
                        cl.SVM(c=10, k=1, ker='Radial', paramker=[0.05])
                    ]),
                    StandardScaler(),
                    cl.LogisticRegression(norm_coeff=0.1)
                    ])
    pip.fit(train, train_labels)
    comp_labels = pip.predict(test, return_prob=True)[1]

    pred1 = np.where(comp_labels > thresh1, True, False)
    conf1 = getConfusionMatrix2(pred1, test_labels)
    r = normalizedBayesRisk(conf1, Cfn=1, Cfp=1, pi1=.5)
    print("DCF: ", r)
    fnr = conf1[0, 1] / (conf1[0, 1] + conf1[1, 1])
    fpr = conf1[1, 0] / (conf1[1, 0] + conf1[0, 0])
    print("fnr, fpr: ", fnr, fpr)
    print("Error rate: ", err_rate(pred1, test_labels))

    pred2 = np.where(comp_labels > thresh2, True, False)
    conf2 = getConfusionMatrix2(pred2, test_labels)
    r = normalizedBayesRisk(conf2, Cfn=1, Cfp=1, pi1=.1)
    print("DCF: ", r)

    pred3 = np.where(comp_labels > thresh3, True, False)
    conf3 = getConfusionMatrix2(pred3, test_labels)
    r = normalizedBayesRisk(conf3, Cfn=1, Cfp=1, pi1=.9)
    print("DCF: ", r)


if __name__ == "__main__":
    test_final()

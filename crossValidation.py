import numpy as np
from sklearn.svm import SVC

#two fold cross validation with linear kernel
def linearTwoFold(X, y, C):
    n, d = X.shape

    # split positive and negative valued samples
    positive_samples = list(np.where(y==1)[0])
    negative_samples = list(np.where(y==-1)[0])

    # split samples into two folds with similar proportion of positive:negative
    samples_in_fold1 = positive_samples[:106] + negative_samples[:178]
    samples_in_fold2 = positive_samples[106:] + negative_samples[178:]


    y_pred = np.zeros(n, int)

    # train with fold1, predict fold2
    alg = SVC(C=C, kernel='linear')
    alg.fit(X[samples_in_fold1], y[samples_in_fold1])
    y_pred[samples_in_fold2] = alg.predict(X[samples_in_fold2])

    # train with fold2, predict fold1
    alg = SVC(C=C, kernel='linear')
    alg.fit(X[samples_in_fold2], y[samples_in_fold2])
    y_pred[samples_in_fold1] = alg.predict(X[samples_in_fold1])

    err = np.mean(y != y_pred)
    print("Two Fold Error:\t%f" % err)

import numpy as np
from sklearn.svm import SVC

#two fold cross validation with linear kernel
def linearTwoFold(X, y, C, kernel):
    n, d = X.shape

    # split positive and negative valued samples
    positive_samples = list(np.where(y==1)[0])
    negative_samples = list(np.where(y==-1)[0])

    # randomize the data for testing
    np.random.shuffle(positive_samples)
    np.random.shuffle(negative_samples)

    # split samples into two folds with similar proportion of positive:negative
    samples_in_fold1 = positive_samples[:106] + negative_samples[:178]
    samples_in_fold2 = positive_samples[106:] + negative_samples[178:]


    y_pred = np.zeros(n, int)

    # train with fold1, predict fold2
    alg = SVC(C=C, kernel=kernel)
    alg.fit(X[samples_in_fold1], y[samples_in_fold1])
    y_pred[samples_in_fold2] = alg.predict(X[samples_in_fold2])

    # train with fold2, predict fold1
    alg = SVC(C=C, kernel=kernel)
    alg.fit(X[samples_in_fold2], y[samples_in_fold2])
    y_pred[samples_in_fold1] = alg.predict(X[samples_in_fold1])

    err = np.mean(y != y_pred)
    print("Two Fold Error:\t%f" % err)

# bootstrapping within two-fold crossValidation
def nestedValidation(X, y, kernel):
    C_list = [0.1, 1.0, 10.0]
    B = 30

    # split positive and negative valued samples
    positive_samples = list(np.where(y==1)[0])
    negative_samples = list(np.where(y==-1)[0])

    # randomize data for testing
    np.random.shuffle(positive_samples)
    np.random.shuffle(negative_samples)

    # split samples into two folds with similar proportion of positive:negative
    samples_in_fold1 = positive_samples[:106] + negative_samples[:178]
    samples_in_fold2 = positive_samples[106:] + negative_samples[178:]

    y_pred = np.zeros(len(X), int)

    best_err = 1.1  # any value greater than 1.0
    best_C = 0.0

    for C in C_list:
        err = bootstrapping(B, X[samples_in_fold1], y[samples_in_fold1], C, kernel)
        print("C=", C, "err=", err)
        if (err <= best_err):
            best_err = err
            best_C = C

    print("Best C=", best_C)

    alg = SVC(C=best_C, kernel=kernel)
    alg.fit(X[samples_in_fold1], y[samples_in_fold1])
    y_pred[samples_in_fold2] = alg.predict(X[samples_in_fold2])

    best_err = 1.1  # any value greater than 1.0
    best_C = 0.0

    for C in C_list:
        err = bootstrapping(B, X[samples_in_fold2], y[samples_in_fold2], C, kernel)
        print("C=", C, "err=", err)
        if (err <= best_err):
            best_err = err
            best_C = C

    print("Best C=", best_C)


    alg = SVC(C=best_C, kernel=kernel)
    alg.fit(X[samples_in_fold2], y[samples_in_fold2])
    y_pred[samples_in_fold1] = alg.predict(X[samples_in_fold1])

    err = np.mean(y != y_pred)

    print("Nested Validation Error=", err)
    return err

def bootstrapping(B, X_subset, y_subset, C, kernel):
    n = len(X_subset)
    bs_err = np.zeros(B)

    for b in range(B):
        train_samples = list(np.random.randint(0,n,n))
        test_samples = list(set(range(n)) - set(train_samples))

        alg = SVC(C=C, kernel=kernel)
        alg.fit(X_subset[train_samples], y_subset[train_samples])
        bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
    err = np.mean(bs_err)

    return err

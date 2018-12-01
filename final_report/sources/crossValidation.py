import numpy as np
from sklearn.svm import SVC

def linearTwoFold(X, y, C, kernel):
    """
    Trains a SVM and tests its accuracy with two-fold cross validation.
    
    X is a numpy matrix containint the predictors for each sample
    y is a numpy vector containing the label for each sample
        y[i] contains the label for X[i]
    C is the slack parameter for the SVM
    kernel is a string for the kernel type of the SVM 
    """
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
    return y_pred

def bootstrapping(B, X_subset, y_subset, C, kernel):
    n = len(X_subset)
    bs_err = np.zeros(B)

    for b in range(B):
        # get n samples (with replacement) from the data
        train_samples = list(np.random.randint(0,n,n))
        test_samples = list(set(range(n)) - set(train_samples))

        # train and test the SVM with given slack C and kernel
        alg = SVC(C=C, kernel=kernel)
        alg.fit(X_subset[train_samples], y_subset[train_samples])
        bs_err[b] = np.mean(y_subset[test_samples] != alg.predict(X_subset[test_samples]))
    err = np.mean(bs_err)

    return err

def nestedValidation(X, y, kernel):
    """
    Performs nested two-fold cross validation with bootstrapping
    to determine an appropriate slack variable for the SVM.
    
    X is a numpy matrix containing the data for each sample
    y is a numpy vector containing the label for each sample
        y[i] contains the label for X[i]
    kernel is a string with the kernel type used for the SVM
    """

    # list of slack variable values to test the SVM with
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # number of iterations to run the bootstraping
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

    # finding the best slack value and its corresponding error
    best_err = 1.0
    best_C = 0.0

    print("testing with first fold")
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

    print("testing with second fold")
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

def nestedKFoldValidation(k, X, y, kernel):
    """
    Performs a nested validation with k-fold validation 
    to determine the best slack value for the support vector machine.

    k is the number of folds to use
    X is a numpy matrix containing the data for each sample
    y is a numpy vector containing the label for each sample
        y[i] contains the label for X[i]
    kernel is the kernel used for the support vector machine
    """

    print("running k-folds cross validation to find best slack C")

    # list of slack variable values used to test the SVM with
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000]
    # number of iterations to run the bootstrapping
    B = 30

    sample_list = list(range(len(y)))
    np.random.shuffle(sample_list)

    # splitting list into k folds
    foldSize = len(y)//k
    folds = [sample_list[i*foldSize:(i+1)*foldSize] for i in range(k)]

    # add remaining items to the last fold
    folds[-1] += sample_list[foldSize*k:]

    test_results = {}
    for test in folds:
        print('==== next outer fold ====')
        # python dictionary to keep track of errors
        error_dict = {}
        for validation in folds:
            # do not want identical test and validation folds
            if validation == test:
                continue

            # training data is all folds except for test and validation
            train = [f for f in folds if f != test and f != validation][0]
            # find errors across all parameters
            for c in C_list:
                alg = SVC(C=c, kernel=kernel)
                alg.fit(X[train], y[train])
                err = np.mean(y[validation] != alg.predict(X[validation]))
                # keep track of all errors
                if c not in error_dict:
                    error_dict[c] = {}
                    error_dict[c]['err'] = []
                    error_dict[c]['model'] = alg
                error_dict[c]['err'].append(err)

        # finding the best parameter from the inner loop
        bestC = -1
        bestError = 1.0
        for c in error_dict:
            print('c=', c, 'err=', np.mean(error_dict[c]['err']))
            err = np.mean(error_dict[c]['err'])
            if err < bestError:
                bestError = err
                bestC = c
        print('best c from inner loop is', bestC)

        # testing the best model from the inner loop against test data
        train_err = np.mean(y[test] != alg.predict(X[test]))
        if bestC not in test_results:
            test_results[bestC] = []
        test_results[bestC].append(train_err)

    for c in test_results:
        print('c=', c,
              'mean error=', np.mean(test_results[c]),
              'stddev of errors=', np.std(test_results[c]))

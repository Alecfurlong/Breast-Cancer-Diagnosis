import numpy as np
import progress
from sklearn.svm import SVC

#leave one out cross validation, returns array of predictions after running svm
def run(X,y,C,kernel):
    n = len(X)
    y_pred = np.zeros(n,int)
    
    for i in range(n):
        all_except_i = list(range(i)) + list(range(i+1,n))
        X_train = X[all_except_i]
        y_train = y[all_except_i]
        alg = SVC(C=C, kernel=kernel)
        alg.fit(X_train, y_train)
        loner = X[i]
        loner = loner.reshape(1, -1)
        y_pred[i] = alg.predict(loner)
        #print (y_pred[i])

        #progress bar
        progress.bar(i+1,n,"Performing Leave One Out Cross Validation")

    err = np.mean(y!=y_pred)
    print ("LEAVE ONE OUT: err=", err)
    return y_pred

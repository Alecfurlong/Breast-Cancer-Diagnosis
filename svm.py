from sklearn.svm import SVC
import numpy as np

import processData

# training and testing on the entire dataset gives 3.3% error
def linearSVM(X, y, C):
    alg = SVC(C=C, kernel='linear')
    alg.fit(X, y)
    y_pred = alg.predict(X)
    err = np.mean(y != y_pred)
    print("Whole Dataset linearSVM Error:\t%f" % err)

from sklearn.svm import SVC
import numpy as np

import processData

X, y, F, ID_list = processData.clean('./raw_breast_cancer_data.csv')

# training and testing on the entire dataset gives 3.3% error

alg = SVC(C=1.0, kernel='linear')
alg.fit(X, y)
y_pred = alg.predict(X)
err = np.mean(y != y_pred)
print(err)

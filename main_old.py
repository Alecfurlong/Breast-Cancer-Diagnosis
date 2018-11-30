import processData
import numpy as np
import matplotlib.pyplot as pp
import crossValidation as cv
import svm

X, y, F, id_list = processData.clean("raw_breast_cancer_data.csv")

# remove 'diagnosis' column
F = F[1:]
print("F:")
print(F)
print(F[23])

print(X.shape)
# counts of types of cancer
print("malignant count: " + str(np.sum(y == 1)))
print("benign count: " + str(np.sum(y == -1)))

positive = list(np.where(y==1)[0])
negative = list(np.where(y==-1)[0])

# malignant and benign on scatter plot of col1 vs col2
"""
pp.figure()
pp.plot(X[negative, 1], X[negative, 2], 'bo')
pp.plot(X[positive, 1], X[positive, 2], 'ro')
pp.show()
"""

# means and stdevs of each column
"""
m = np.mean(X, axis=0)
s = np.std(X, axis=0)
print(m.size)
print(s.size)

pp.figure()
pp.plot(m, 'b+')

pp.figure()
pp.plot(s, 'ro')

pp.show()
"""

# correlation heatmap
"""
pp.matshow(np.corrcoef(X, rowvar=False))
pp.show()
"""

# run linear SVM with C=1.0
svm.linearSVM(X, y, 1.0)

# two fold cross validation, C=1.0, linear kernel
cv.linearTwoFold(X, y, 1.0, 'linear')

# bootstrapping within two-fold crossValidation, linear kernel
cv.nestedValidation(X, y, 'linear')

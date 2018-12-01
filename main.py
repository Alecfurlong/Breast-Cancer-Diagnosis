import processData
import numpy as np
import matplotlib.pyplot as pp
import crossValidation as cv
import svm

X, y, F, id_list = processData.clean("raw_breast_cancer_data.csv")

# remove 'diagnosis' column
F = F[1:]
# print data column headers
# print("F:")
# print(F)
# print(F[23])

# counts of types of cancer
print("malignant count: " + str(np.sum(y == 1)))
print("benign count: " + str(np.sum(y == -1)))

cv.nestedKFoldValidation(10, X, y, 'linear')

# two fold cross validation, C=1.0, linear kernel
# cv.linearTwoFold(X, y, 1.0, 'linear')

# bootstrapping within two-fold crossValidation, linear kernel
# cv.nestedValidation(X, y, 'linear')

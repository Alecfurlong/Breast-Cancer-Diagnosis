# imported libraries
import numpy as np
import matplotlib.pyplot as pp

# group project code
import crossValidation as cv
import processData
import diagnostics

X, y, F, id_list = processData.clean("raw_breast_cancer_data.csv")

# remove 'diagnosis' column from column labels
F = F[1:]
print(F)

# deleting highly correlated predictor variables where |correlation| > 0.9
#F, X = diagnostics.deleteColumns([2, 3, 12, 13, 20, 21, 22, 23, 24], F, X)

# counts of types of cancer
print("malignant count: " + str(np.sum(y == 1)))
print("benign count: " + str(np.sum(y == -1)))
cv.nestedValidation(X, y, 'rbf')

cv.nestedKFoldValidation(10, X, y, 'rbf')

# two fold cross validation, C=1.0, linear kernel
# cv.linearTwoFold(X, y, 1.0, 'linear')

# bootstrapping within two-fold crossValidation, linear kernel
# cv.nestedValidation(X, y, 'linear')

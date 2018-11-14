import processData 
import numpy as np
import matplotlib.pyplot as pp

X, y, F, id_list = processData.clean("raw_breast_cancer_data.csv")

print(X.shape)
# counts of types of cancer
print("malignant count: " + str(np.sum(y == 1)))
print("benign count: " + str(np.sum(y == -1)))

positive = list(np.where(y==1)[0])
negative = list(np.where(y==-1)[0])

pp.figure()
pp.plot(X[negative, 1], X[negative, 2], 'bo')
pp.plot(X[positive, 1], X[positive, 2], 'ro')
pp.show()

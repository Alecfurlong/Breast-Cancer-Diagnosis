import processData
import numpy as np
import matplotlib.pyplot as pp
import crossValidation as cv
import svm

# removing highly correlated attributes
def deleteColumn(n, F, X):
    X = np.delete(X, n, 1)
    F = F[:n] + F[n:]
    return F, X

def deleteColumns(n, F, X):
    n.sort(reverse=True)
    print(n)
    for i in n:
        F, X = deleteColumn(i, F, X)
    return F, X

def main():
    # loading the raw data
    X, y, F, id_list = processData.clean("raw_breast_cancer_data.csv")

    # remove 'diagnosis' column
    # F = F[1:]
    del F[1]
    print("F:")
    print(F)

    # correlation heatmap before removing highly correlated columns
    pp.matshow(np.corrcoef(X, rowvar=False))
    pp.show()

    F, X = deleteColumns([2, 3, 12, 13, 20, 21, 22, 23, 24], F, X)

    # correlation heatmap after removing highly correlated columns
    pp.matshow(np.corrcoef(X, rowvar=False))
    pp.show()

    # counts of types of cancer
    print("malignant count: " + str(np.sum(y == 1)))
    print("benign count: " + str(np.sum(y == -1)))

    positive = list(np.where(y==1)[0])
    negative = list(np.where(y==-1)[0])

    # malignant and benign on scatter plot of col1 vs col2
    pp.figure()
    pp.plot(X[negative, 1], X[negative, 2], 'bo')
    pp.plot(X[positive, 1], X[positive, 2], 'ro')
    pp.show()

    # means and stdevs of each column
    m = np.mean(X, axis=0)
    s = np.std(X, axis=0)
    print(m.size)
    print(s.size)

    pp.figure()
    pp.plot(m, 'b+')

    pp.figure()
    pp.plot(s, 'ro')

    pp.show()

if __name__ == '__main__':
    main()

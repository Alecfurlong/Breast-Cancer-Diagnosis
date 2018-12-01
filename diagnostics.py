import processData
import numpy as np
import matplotlib.pyplot as pp
import crossValidation as cv
import svm
from sklearn.svm import SVC

# removing highly correlated attributes
def deleteColumn(n, F, X):
    X = np.delete(X, n, 1)
    F = F[:n] + F[n:]
    return F, X

def deleteColumns(n, F, X):
    removing = [F[i] for i in n]
    print('deleting columns: ', removing)
    n.sort(reverse=True)
    for i in n:
        F, X = deleteColumn(i, F, X)
    return F, X

def drawParameterVAccuracyPlot():
    X, y, F, id_list = processData.clean("raw_breast_cancer_data.csv")
    
    C_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000]
    # C_list = [0.001, 0.01, 0.1]

    n = len(y)
    data = [x for x in range(n)]
    np.random.shuffle(data)

    train = data[:int(n*0.8)]
    test = data[int(n*0.8):]

    errors = {}
    for c in C_list:
        print(c)
        alg = SVC(C=c, kernel='linear')
        alg.fit(X[train], y[train])
        errors[c] = np.mean(y[test] != alg.predict(X[test]))

    y_pos = np.arange(len(C_list))
    errList = [errors[c] for c in C_list]
    pp.bar(y_pos, errList, align='center')
    pp.xticks(y_pos, C_list)
    pp.xlabel('Slack Value')
    pp.ylabel('Error Rate')
    pp.title('Parameter VS Accuracy')
    pp.show()

def main():
    drawParameterVAccuracyPlot()
    return

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

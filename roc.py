import numpy as np
import leaveOneOut as loocv
import crossValidation as cv
import matplotlib.pyplot as pp
from sklearn import metrics
import processData

def generateROC(X,y):
    #get true positive rate, false positive rate
    tpr, fpr = get_true_false_rates(X,y)
    
    pp.figure()
    pp.xlabel("False Positive Rate")
    pp.ylabel("True Positive Rate")
    pp.title("ROC Curve")

    #calc area under curve and plot
    area_under_curve = metrics.auc(fpr, tpr)
    roc_label = '{} (AUC={:.3f})'.format('Method', area_under_curve)
    pp.plot(fpr, tpr, color='orange', linewidth=2, label=roc_label)

    #plot linear line
    x = [0.0, 1.0]
    pp.plot(x, x, linestyle='dashed', color='blue', linewidth=2, label='random')
    pp.xlim(0.0, 1.0)
    pp.ylim(0.0, 1.0)
    pp.legend(fontsize=10, loc='best')
    pp.tight_layout()

    pp.show()

def get_true_false_rates(X,y):
    tpr = [0.0]  # true positive rate
    fpr = [0.0]  # false positive rate
    
    #real values
    malignantCount = float(np.sum(y == 1))
    benignCount = float(np.sum(y == -1))

    truePositives = 0
    falsePositives = 0
    correctlyClassifiedMal = 0.0
    correctlyClassifiedBen = 0.0

    #get predictions using leave one out cross validation
    y_pred = loocv.run(X, y, 0.01, 'linear')

    for i, ypredi in enumerate(y_pred):
        # only worry about predicted positives
        if ypredi == 1:
            # true positive
            if y[i] == 1:
                truePositives += 1
            # false positive
            else:
                falsePositives += 1

        tpr.append(truePositives / malignantCount)
        fpr.append(falsePositives / benignCount)

    tpr.append(1.0)
    fpr.append(1.0)

    return tpr, fpr

def main():
    X, y, F, ids = processData.clean('raw_breast_cancer_data.csv')
    generateROC(X, y)

if __name__ == '__main__':
    main()

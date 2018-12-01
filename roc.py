import numpy as np
import leaveOneOut as loocv
import crossValidation as cv
import matplotlib.pyplot as pp
from sklearn import metrics

def generateROC(X,y):
	#init graph
	pp.figure()
	pp.xlabel("False Positive Rate")
	pp.ylabel("True Positive Rate")
	pp.title("ROC Curve")

	#plot line 1 ###########################
	y_pred = loocv.run(X,y,0.0001,'linear')
	#get true positive rate, false positive rate
	tpr, fpr = get_true_false_rates(X,y,y_pred)
	
	#calc area under curve and plot
	area_under_curve = metrics.auc(fpr, tpr)
	roc_label = '{} (AUC={:.3f})'.format('Linear SVM w/ LOOCV', area_under_curve)
	pp.plot(fpr, tpr, color='orange', linewidth=2, label=roc_label)

	#plot line 2 ###########################
	y_pred = cv.linearTwoFold(X, y, 1.0, 'linear')
	#get true positive rate, false positive rate
	tpr, fpr = get_true_false_rates(X,y,y_pred)
	
	area_under_curve = metrics.auc(fpr, tpr)
	roc_label = '{} (AUC={:.3f})'.format('Linear SVM w/ 2-Fold', area_under_curve)
	pp.plot(fpr, tpr, color='red', linewidth=2, label=roc_label)


	#plot linear line
	x = [0.0, 1.0]
	pp.plot(x, x, linestyle='dashed', color='blue', linewidth=2, label='random')
	pp.xlim(0.0, 1.0)
	pp.ylim(0.0, 1.0)
	pp.legend(fontsize=10, loc='best')
	pp.tight_layout()
	
	pp.show()
	pass

def get_true_false_rates(X,y, y_pred):
	tpr = [0.0]  # true positive rate
	fpr = [0.0]  # false positive rate
	
	#real values
	malignantCount = float(np.sum(y == 1))
	benignCount = float(np.sum(y == -1))

	correctlyClassifiedMal = 0.0
	correctlyClassifiedBen = 0.0

	for i in range(len(X)):
		#if y[i] matches prediction for X[i]
		if y[i] == y_pred[i]:
			if y[i] == 1:
				#correctly classified Malignant
				correctlyClassifiedMal += 1.0
			else:
				#correctly classified Malignant
				correctlyClassifiedBen += 1.0
		
		tpr.append(correctlyClassifiedMal / malignantCount)
		fpr.append(correctlyClassifiedBen / benignCount)

	return tpr, fpr

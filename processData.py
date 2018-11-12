import numpy as np
#convert raw csv file to matrix, convert M(Malignant) to +1 and B(Benign) to -1
#initialize array to full size of breast cancer data
#569 samples, 32 features, feature 0 is breast cancer id number

def clean(filepath):
	X = np.zeros((569,31))			#Data Matrix
	ID_dict = []								#ID dictionary, ID[0] corresponds to Matrix[0]
	F = []											#Feature Definitions				

	with open(filepath) as fp:
		#first line is column definitions
		line = fp.readline()
		definitions = line.split(',')
		for n in range(len(definitions)-2):
			F.append(definitions[n+1])

		line = fp.readline()		
		i = 0
		while line:
			#get features
			features = line.split(',')
			
			#get id and append
			ID_dict.append(features[0])
			
			#convert M to +1, B to -1
			if features[1] is 'M':
				features[1] = 1.0
			else:
				features[1] = -1.0

			#add features to matrix
			for j in range(len(features)-1):
				X[i][j] = features[j+1]
			
			#get next line
			line = fp.readline()
			i += 1

	#check if X has any Nan
	if np.any(np.isnan(X)):
		"Data is corrupted, contains Nan"
		exit(1)
	
	return X, F, ID_dict;

import numpy as np
import csv

def clean(filepath):
    rawtable = []

    # load entire csv into rawtable
    with open(filepath) as fp:
        reader = csv.reader(fp)
        for row in reader:
            rawtable.append(row)
    
    # F is the list of features (first row of the table)
    # rawtable[0][1:] -> "get all columns after the first one in the first row"
    F = rawtable[0][1:]

    # ID_list is the ID of each sample (first column of every row)
    ID_list = [int(sample[0]) for sample in rawtable[1:]]

    # X is the numpy array of all X values
    X = [sample[1:] for sample in rawtable[1:]]
    for i, sample in enumerate(X):
        # +1 for M, -1 for B
        X[i][0] = 1.0 if sample[0] == 'M' else -1.0

    # cast to floats
    X = np.array(X).astype(float)

    return X, F, ID_list

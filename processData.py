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
    data = [sample[1:] for sample in rawtable[1:]]
    for i, sample in enumerate(data):
        # +1 for M, -1 for B
        data[i][0] = 1.0 if sample[0] == 'M' else -1.0

    # cast to floats
    data = np.array(data).astype(float)
    y = data[:, 0]
    X = data[:, 1:]

    return X, y, F, ID_list

# Breast-Cancer-Diagnosis

The main python file runs process data.

raw_breast_cancer_data.csv is the untouched data set downloaded from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

processData.py converts the csv to a workable data matrix, column definition list, and an id-list

matrix dimensions of the cleaned data are 569 by 31, removed first row because it contained definitions such as "id, perimeter" and removed first column of
features since an id number: 80290 for example should not be included in calculations

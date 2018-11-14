import processData 

X, F, id_list = processData.clean("raw_breast_cancer_data.csv")

print("id -list")
print(id_list)
print()

print("Feature Definitions")
print(F)
print()

print("Data")
print(X)

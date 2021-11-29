from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#loading Datasets:
ires = datasets.load_iris()

#Printing dicription and Features:
#print(ires.DESCR)
features = ires.data
label = ires.target

print("features:", features[0], "\n label:", label[0])

#Train the classifier:
clr = KNeighborsClassifier()
clr.fit(features, label)

pred = clr.predict([[2, 8, 3, 9]])
print(pred)
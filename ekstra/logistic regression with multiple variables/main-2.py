import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

dataset = pd.read_csv("heart-train.csv")
testset = pd.read_csv("heart-test.csv")

x = dataset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
y = dataset.iloc[:, 13].values

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

print("Coeffcients : \n",classifier.coef_)
print("Intercept : \n",classifier.intercept_)

y_pred = classifier.predict(xtest)
cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)

print ("Accuracy : ", accuracy_score(ytest, y_pred))



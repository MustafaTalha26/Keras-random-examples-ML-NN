import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

trainset = pd.read_csv("heart-train.csv")
testset = pd.read_csv("heart-test.csv")

xtrain = trainset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
ytrain = trainset.iloc[:, 13].values

xtest = testset.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,12]].values
ytest = testset.iloc[:, 13].values

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

heart_age = trainset.iloc[:, [0,13]].values
heart_age_positive = [f for f in heart_age if f[1] == 1]
heart_age_negative = [f for f in heart_age if f[1] == 0]

heart_chol = trainset.iloc[:, [4,13]].values
heart_chol_positive = [f for f in heart_chol if f[1] == 1]
heart_chol_negative = [f for f in heart_chol if f[1] == 0]

heart_fbs = trainset.iloc[:, [5,13]].values
heart_fbs_positive = [f for f in heart_fbs if f[1] == 1]
heart_fbs_negative = [f for f in heart_fbs if f[1] == 0]

heart_oldpeak = trainset.iloc[:, [9,13]].values
heart_oldpeak_positive = [f for f in heart_oldpeak if f[1] == 1]
heart_oldpeak_negative = [f for f in heart_oldpeak if f[1] == 0]

heart_age_negative = pd.DataFrame(np.array(heart_age_negative).reshape(-2,2))[0]
heart_age_positive = pd.DataFrame(np.array(heart_age_positive).reshape(-2,2))[0]

heart_chol_negative = pd.DataFrame(np.array(heart_chol_negative).reshape(-2,2))[0]
heart_chol_positive = pd.DataFrame(np.array(heart_chol_positive).reshape(-2,2))[0]

heart_fbs_negative = pd.DataFrame(np.array(heart_fbs_negative).reshape(-2,2))[0]
heart_fbs_positive = pd.DataFrame(np.array(heart_fbs_positive).reshape(-2,2))[0]

heart_oldpeak_negative = pd.DataFrame(np.array(heart_oldpeak_negative).reshape(-2,2))[0]
heart_oldpeak_positive = pd.DataFrame(np.array(heart_oldpeak_positive).reshape(-2,2))[0]

fig, axs = plt.subplots(2,4, sharey=True, tight_layout=True)

colors = ['aqua','deepskyblue','red','tan','bisque','orchid','teal','slateblue']
axs[0][0].hist(heart_age_negative,histtype='bar', color=colors[0], ec='black', label='Ages-', bins=10)
axs[1][0].hist(heart_age_positive,histtype='bar', color=colors[1], ec='black', label='Ages+', bins=10)
axs[0][0].legend(prop={'size': 10})
axs[1][0].legend(prop={'size': 10})

axs[0][1].hist(heart_chol_negative,histtype='bar', color=colors[2], ec='black', label='Chol-', bins=10)
axs[1][1].hist(heart_chol_positive,histtype='bar', color=colors[3], ec='black', label='Chol+', bins=10)
axs[0][1].legend(prop={'size': 10})
axs[1][1].legend(prop={'size': 10})

axs[0][2].hist(heart_fbs_negative,histtype='bar', color=colors[4], ec='black', label='Fbs-', bins=10)
axs[1][2].hist(heart_fbs_positive,histtype='bar', color=colors[5], ec='black', label='Fbs+', bins=10)
axs[0][2].legend(prop={'size': 10})
axs[1][2].legend(prop={'size': 10})

axs[0][3].hist(heart_oldpeak_negative,histtype='bar', color=colors[6], ec='black', label='Oldpeak-', bins=10)
axs[1][3].hist(heart_oldpeak_positive,histtype='bar', color=colors[7], ec='black', label='Oldpeak+', bins=10)
axs[0][3].legend(prop={'size': 10})
axs[1][3].legend(prop={'size': 10})
plt.show()






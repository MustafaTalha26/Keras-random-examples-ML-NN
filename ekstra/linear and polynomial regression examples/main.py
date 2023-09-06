import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from pandas import read_csv
import csv
with open("income.data.csv", 'r') as f:
  data = list(csv.reader(f, delimiter=","))

data = np.array(data, dtype=(np.float16, np.float16))
data = data.reshape(2,-2)

x = np.array(data[0]).reshape(-1,1)
y = np.array(data[1])

model = LinearRegression().fit(x, y)
y_pred = model.predict(x)

plt.plot(x, y, 'o')
plt.plot(x, y_pred, '.')
plt.show()

###########################################################################

x_ = PolynomialFeatures(degree=4, include_bias=False).fit_transform(x)
model2 = LinearRegression().fit(x_, y)
y_pred = model2.predict(x_)

plt.plot(x, y, 'o')
plt.plot(x, y_pred, '.')
plt.show()



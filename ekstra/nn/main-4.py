import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import pandas as pd

def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(64, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mae', optimizer='adam')
	return model

dataset = pd.read_csv('ENB2012_data.csv')

X = dataset.drop(['Y1', 'Y2'], axis=1).values
y = (dataset[['Y1', 'Y2']]).values

x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size=0.8, random_state=0)

n_inputs, n_outputs = X.shape[1], y.shape[1]
model = get_model(n_inputs, n_outputs)
history = model.fit(x_train, y_train,validation_data=(x_test, y_test), verbose=0, epochs=100)

train_mae = model.evaluate(x_train, y_train, verbose=0)
test_mae = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_mae, test_mae))

plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

y_pred = model.predict(x_test)
y_pred = pd.DataFrame(y_pred, columns = ['Y1','Y2'])
y_test = pd.DataFrame(y_test, columns = ['Y1','Y2'])

y1x = y_pred.iloc[:, 0].values
y1y = y_pred.iloc[:, 1].values 

y2x = y_test.iloc[:, 0].values
y2y = y_test.iloc[:, 1].values

error = mean_squared_error(y_test.iloc[:, 0].values, y_pred.iloc[:, 0].values, squared=False)
print("RMSE1: " + str("{:.2f}".format(error)))

error2 = mean_squared_error(y_test.iloc[:, 1].values, y_pred.iloc[:, 1].values, squared=False)
print("RMSE2: " + str("{:.2f}".format(error2)))

plt.scatter(y1x, y1y, c='green')
plt.scatter(y2x, y2y, c='red')
plt.show()

x1 = pd.DataFrame(x_test).iloc[:, 7].values

plt.plot(x1, y2x,'.')
plt.plot(x1, y1x,'.')
plt.show()

guess = pd.DataFrame(y_test, columns = ['Y1','Y2'])
guess['Y3'] = y1x
guess['Y4'] = y1y
print(guess)


import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import date
from datetime import timedelta

from FinMind.data import DataLoader

'''
大部分參考medium那篇 另外再改一些數值
'''

dl = DataLoader()
data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2018-01-01')

data_df.to_csv("lstm_data.csv")
data_df = pd.read_csv("lstm_data.csv")
data_df.set_index("date", inplace=True)
'''
data_df['close'].plot()
plt.ylabel("close price")
plt.show()
'''

# prepare training data
close_prices = data_df['close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

# use 60 days before to predict 60 days after 
window_size = 60
x_train = []
y_train = []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# prepare test data
test_data = scaled_data[training_data_len-window_size: , : ]
x_test = []
y_test = values[training_data_len:]

for i in range(window_size, len(test_data)):
  x_test.append(test_data[i-window_size:i, 0])

# setting up LSTM 
# not sure how this work
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
model = keras.Sequential()
model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(50, return_sequences=False))
model.add(keras.layers.Dense(25))
model.add(keras.layers.Dense(1))
model.summary()

# train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size= 32, epochs=100)
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = 0.0
for i in range(len(predictions)):
  rmse += (predictions[i]-y_test[i])**2
rmse = math.sqrt((rmse/len(predictions)))
print(rmse)

# plot traing result
data = data_df.filter(['close'])
train = data[:training_data_len]
validation = data[training_data_len:]
validation['Predictions'] = predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.plot(train)
plt.plot(validation[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

# prediction for next 60 days
predict_data = scaled_data[len(scaled_data)-window_size:, :]
predict_data = np.array(predict_data)
predict_data = np.reshape(predict_data, (predict_data.shape[0], predict_data.shape[1], 1))
predictions = model.predict(predict_data)
predictions = scaler.inverse_transform(predictions)
fdate = []
skip = 0
fdate.append(date.today())
for i in range(1, window_size):
  temp = date.today()
  temp += timedelta(i+skip)
  if temp.weekday() == 5:
    skip += 2
    temp += timedelta(2)
  fdate.append(temp)

plt.figure(figsize=(16,8))
plt.plot(fdate, predictions)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
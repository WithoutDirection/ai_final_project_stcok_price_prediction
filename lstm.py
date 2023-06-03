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
之後再打註解
'''

'''
data_df['close'].plot()
plt.ylabel("close price")
plt.show()
'''
class Lstm():
  def __init__(self):
    self.model = keras.Sequential()
    self.window_size = 60
    self.scaler = None
    self.scaled = None
    self.actual_scaled = None
    self.closed = None
    self.actual_closed = None
  def train(self, close_prices):
    self.actual_closed = close_prices
    values = self.actual_closed.values
    training_data_len = len(values)-14
    self.scaler = MinMaxScaler(feature_range=(0,1))
    self.actual_scaled = self.scaler.fit_transform(values.reshape(-1,1))
    train_data = self.actual_scaled[0: training_data_len, :]

    self.closed = close_prices[0:training_data_len]
    self.scaled = train_data

    x_train = []
    y_train = []
    for i in range(self.window_size, len(train_data)):
        x_train.append(train_data[i-self.window_size:i, 0])
        y_train.append(train_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    test_data = self.actual_scaled[training_data_len-self.window_size: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(self.window_size, len(test_data)):
      x_test.append(test_data[i-self.window_size:i, 0])

    # setting up LSTM 
    # not sure how this work
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    self.model.add(keras.layers.LSTM(32, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    self.model.add(keras.layers.LSTM(32, return_sequences=True))
    self.model.add(keras.layers.LSTM(32))
    self.model.add(keras.layers.Dense(1))
    self.model.summary()

    self.model.compile(optimizer='adam', loss='mean_squared_error')
    self.model.fit(x_train, y_train, batch_size=32, epochs=100)
    predictions = self.predict(14, train=True)
    self.rsme(predictions, y_test)


    data = data_df.filter(['close'])
    train = data[:training_data_len]
    test = data[training_data_len-1:]
    predictions.insert(0, data.iloc[training_data_len-1, 0])
    test['Predictions'] = predictions
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.plot(train)
    plt.plot(test[['close', 'Predictions']])
    plt.legend(['Train', 'Test', 'Predictions'], loc='lower right')
    plt.show()    
      
    self.actual_data()
   
  def rsme(self, predictions, y_test):
    rsme = 0.0
    for i in range(len(predictions)):
      rsme += (predictions[i]-y_test[i])**2
    rsme /= len(predictions)
    rsme = math.sqrt(rsme)
    print("Root Square Mean Error: ", rsme)
    return 
    
  def actual_data(self):
    self.scaled = self.actual_scaled
    self.closed = self.actual_closed
    
  def predict(self, days, train):
    predictions = []
    fdate = []
    #day = datetime.datetime.strptime('2023/05/14', "%Y/%m/%d")
    day = date.today()
    for i in range(1, days+1):
      day += timedelta(1)
      if day.weekday() == 5:
        day += timedelta(2)
      fdate.append(day)
      predict_data = []
      predict_data.append(self.scaled[len(self.scaled)-self.window_size:, 0])
      predict_data = np.array(predict_data)
      predict_data = np.reshape(predict_data, (predict_data.shape[0], predict_data.shape[1], 1))
      prediction = self.model.predict(predict_data)
      prediction = self.scaler.inverse_transform(prediction)
      prediction = round(prediction[0][0], 2)
      temp = pd.Series([prediction])
      self.closed = pd.concat([self.closed, temp], ignore_index=True)
      predictions.append(prediction)
      self.update_scaled_data()
    
    if train is False:
      plt.figure(figsize=(16,8))
      plt.plot(fdate, predictions)
      plt.xlabel('Date')
      plt.ylabel('Close Price')
      plt.show()

    return predictions
  
  def update_scaled_data(self):
    values = self.closed.values
    self.scaled = self.scaler.fit_transform(values.reshape(-1,1))
    
    
if __name__ == "__main__":
  dl = DataLoader()
  data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2018-01-01')
  data_df.to_csv("lstm_data.csv")
  data_df = pd.read_csv("lstm_data.csv")
  data_df.set_index("date", inplace=True)
  close_prices = data_df['close']
  lstm = Lstm()
  lstm.train(close_prices)
  result = lstm.predict(14, train=False)
  print(result)
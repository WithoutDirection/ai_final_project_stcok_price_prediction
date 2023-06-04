import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import date
from datetime import timedelta

from FinMind.data import DataLoader

keras.utils.set_random_seed(12345)

class Lstm():
  def __init__(self):
    self.model = keras.Sequential()
    self.window_size = 60
    self.scaler = None
    self.scaled_data = None
    self.close_prices = None

  # Calculate root square mean error for test data
  def rsme(self, predictions, y_test):
    rsme = 0.0
    for i in range(len(predictions)):
      rsme += (predictions[i]-y_test[i])**2
    rsme /= len(predictions)
    rsme = math.sqrt(rsme)
    print("Root Square Mean Error: ", rsme)
    return rsme
  
  def split_data(self):
    values = self.close_prices.values
    training_data_len = math.ceil(len(values)* 0.8)
    self.scaler = MinMaxScaler(feature_range=(0,1))
    self.scaled_data = self.scaler.fit_transform(values.reshape(-1,1))
    
    train_data = self.scaled_data[0: training_data_len, :]
    x_train = []
    y_train = []
    for i in range(self.window_size, len(train_data)):
        x_train.append(train_data[i-self.window_size:i, 0])
        y_train.append(train_data[i, 0])
        
    test_data = self.scaled_data[training_data_len-self.window_size: , : ]
    x_test = []
    y_test = values[training_data_len:]

    for i in range(self.window_size, len(test_data)):
      x_test.append(test_data[i-self.window_size:i, 0])
    
    return x_train, y_train, x_test, y_test, training_data_len
  
  def train(self, close_prices):
    self.close_prices = close_prices
    x_train, y_train, x_test, y_test, training_data_len = self.split_data()

        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # setting up LSTM 
    self.model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    self.model.add(keras.layers.LSTM(50))
    self.model.add(keras.layers.Dense(1))
    self.model.summary()

    self.model.compile(optimizer='adam', loss='mse')
    self.model.fit(x_train, y_train, batch_size=16, epochs=50)
    predictions = self.model.predict(x_test)
    predictions = self.scaler.inverse_transform(predictions)
    rsme = self.rsme(predictions, y_test)

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
    plt.legend(['Real', 'Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    
    return rsme
    
  def predict(self, days):
    predictions = []
    fdate = []
    day = date.today()
    print(day)
    for _ in range(1, days+1):
      day += timedelta(1)
      if day.weekday() == 5:
        day += timedelta(2)
      fdate.append(day)
      predict_data = []
      predict_data.append(self.scaled_data[len(self.scaled_data)-self.window_size:, 0])
      predict_data = np.array(predict_data)
      predict_data = np.reshape(predict_data, (predict_data.shape[0], predict_data.shape[1], 1))
      prediction = self.model.predict(predict_data)
      prediction = self.scaler.inverse_transform(prediction)
      prediction = round(prediction[0][0], 2)
      temp = pd.Series([prediction])
      self.close_prices = pd.concat([self.close_prices, temp], ignore_index=True)
      predictions.append(prediction)
      self.update_scaled_data()
    
    plt.figure(figsize=(16,8))
    plt.plot(fdate, predictions)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()
    return predictions
  
  def update_scaled_data(self):
    values = self.close_prices.values
    self.scaled_data = self.scaler.fit_transform(values.reshape(-1,1))
    
    
if __name__ == "__main__":
  data_df = pd.read_csv("tsmc_stock_from2020.csv")
  data_df.set_index("date", inplace=True)
  close_prices = data_df['close']
  lstm = Lstm()
  rsme = lstm.train(close_prices)
  result = lstm.predict(14)
  print("Test data root square mean error: ", rsme)
  print("Predict future 14 days closed price: ", result)
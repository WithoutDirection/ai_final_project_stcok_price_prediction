import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from tensorflow import keras
import datetime

keras.utils.set_random_seed(6789)

class Lstm():
  def __init__(self):
    self.model = keras.Sequential()
    self.window_size = 60
    self.scaler = None
    self.scaled_data = None
    self.close_prices = None

  # Train the LSTM model
  def train(self, close_prices):
    self.close_prices = close_prices
    x_train, y_train, x_test, y_test, training_data_len = self.split_data()

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    # setting up LSTM model
    self.model.add(keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    self.model.add(keras.layers.LSTM(50))
    self.model.add(keras.layers.Dense(1))
    self.model.summary()

    self.model.compile(optimizer='adam', loss='mse')
    self.model.fit(x_train, y_train, batch_size=16, epochs=50)
    predictions = self.model.predict(x_test)
    predictions = self.scaler.inverse_transform(predictions)
    rmse, accuracy = self.analyze(predictions, y_test)

    # plot the result of test data
    data = data_df.filter(['close'])
    train = data[:training_data_len]
    validation = data[training_data_len:]
    validation['Predictions'] = predictions
    
    return rmse, accuracy

  # Calculate root square mean error for test data
  def analyze(self, predictions, y_test):
    accuracy = 0
    rmse = 0.0
    for i in range(len(predictions)):
      errors = abs(predictions[i]-y_test[i])
      mape = 100 * (errors / y_test[i])
      accuracy += 100 - np.mean(mape)
      rmse += (predictions[i]-y_test[i])**2
    rmse /= len(predictions)
    rmse = math.sqrt(rmse)
    accuracy /= len(predictions)
    return rmse, accuracy
  
  # split the data into train data and test data the ratio is 8:2
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
   
  # predict the future stock price  
  def predict(self, days):
    predictions = []
    fdate = []
    day = datetime.datetime(2023, 5, 30)
    for _ in range(1, days+1):
      day += datetime.timedelta(1)
      if day.weekday() == 5:
        day += datetime.timedelta(2)
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
    
    result = pd.DataFrame(predictions, index=fdate) 
    plt.xlabel('Date')
    plt.ylabel('Closed')
    plt.plot(result)
    plt.show()
    return result
  
  # update data after a prediction
  def update_scaled_data(self):
    values = self.close_prices.values
    self.scaled_data = self.scaler.fit_transform(values.reshape(-1,1))
    
    
if __name__ == "__main__":
  #dl = DataLoader()
  #data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2022-01-01')
  #data_df.to_csv("2330_data.csv")
  data_df = pd.read_csv("2330_data.csv")
  data_df.set_index("date", inplace=True)
  close_prices = data_df['close']
  lstm = Lstm()
  rmse, accuracy = lstm.train(close_prices)
  result = lstm.predict(14)
  print("Test data root mean square error: ", rmse)
  print(f'Test data accuracy: {round(accuracy, 2)}%')  
  print("Predict future 14 days closed price: ", result)
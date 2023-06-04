import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from FinMind.data import DataLoader

stock_id = '2330'
start_date = '2022-01-01'
window_size = 60
predict_days = 14

def getdata_df(stock_id, start_date):
    dl = DataLoader()
    data_dl = dl.taiwan_stock_daily(stock_id = stock_id, start_date = start_date,end_date='2023-05-25')
    data_dl.to_csv("stock_data.csv")
    data_df = pd.read_csv("stock_data.csv")
    return data_df

def split_xy(window_size, train_data):
    x = []
    y = []
    for i in range(window_size, len(train_data)):
        x.append(train_data.iloc[i-window_size, 5:9])
        for j in range(1,window_size):
            x[i-window_size].loc['open'+str(j)] = train_data.iloc[i-window_size+j, 5]
            x[i-window_size].loc['max'+str(j)] = train_data.iloc[i-window_size+j, 6]
            x[i-window_size].loc['min'+str(j)] = train_data.iloc[i-window_size+j, 7]
            x[i-window_size].loc['close'+str(j)] = train_data.iloc[i-window_size+j, 8]
        y.append(train_data.iloc[i, 5:9])
    print("len of train data:",len(x))
    #print(x[0])
    #print(type(x[0]))
    #print(y[0])
    #print(type(y[0]))
    return x, y

def hyperparameter_tuning(x_train, y_train):
    grid_rf = {
        'n_estimators': [20, 50, 100, 500, 1000],  
        'max_depth': np.arange(1, 10, 1),  
        'min_samples_split': [2, 10, 9], 
        'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
        'bootstrap': [True, False], 
        'random_state': [1, 2, 30, 42]
    }
    model = RandomForestRegressor()
    rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
    rscv_fit = rscv.fit(x_train, y_train)
    best_parameters = rscv_fit.best_params_
    print("best parameter for RF regressor:\n",best_parameters)
    model = RandomForestRegressor(random_state=best_parameters['random_state'],
                              n_estimators=best_parameters['n_estimators'],
                              min_samples_split=best_parameters['min_samples_split'],
                              min_samples_leaf=best_parameters['min_samples_leaf'],
                              max_depth=best_parameters['max_depth'], 
                              bootstrap=best_parameters['bootstrap']) 
    return model

def score(predict, real):
    print("Mean Absolute Error:", round(metrics.mean_absolute_error(real, predict), 4))
    print("Mean Squared Error:", round(metrics.mean_squared_error(real, predict), 4))
    print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(real, predict)), 4))
    print("(R^2) Score:", round(metrics.r2_score(real, predict), 4))
    # print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
    errors = abs(predict - real)
    mape = 100 * (errors / real)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:')
    print(round(accuracy, 2), '%.') 

def prediction(days, model, exist_data, window_size):
    for i in range(days):
        data_x = []
        data_x.append(exist_data.iloc[-window_size, :])
        for k in range(1,window_size):
            data_x[0].loc['open'+str(k)] = exist_data.iloc[-window_size+k, 0]
            data_x[0].loc['max'+str(k)] = exist_data.iloc[-window_size+k, 1]
            data_x[0].loc['min'+str(k)] = exist_data.iloc[-window_size+k, 2]
            data_x[0].loc['close'+str(k)] = exist_data.iloc[-window_size+k, 3]      

        predict_x = model.predict(data_x)
        predict_x = np.round(predict_x, decimals=2)
        #print('predict_x')
        #print(predict_x)
        exist_data.loc[len(exist_data)] = predict_x[0]
        # print(type(predict_x[0]))
    print("***data after prediction:***")
    print(exist_data.iloc[-(days):,:])
    return exist_data.iloc[-(days):, :]

def create_prediction_with_date(new_data, predict_days, last_date):
    next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
    new_dates = pd.date_range(start=next_date, periods=predict_days, freq='B')
    for i in range(predict_days):
        new_data.loc[i, 'date'] = new_dates[i].date()
    return new_data

def plot(data):
    print()


# 以下是main
data_df = getdata_df(stock_id, start_date)
x, y = split_xy(window_size, data_df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,  random_state=0)
print('param tuning...')
# model = hyperparameter_tuning(x_train, y_train) #240個參數tune超久==，得到一次後就自己填:D
model = RandomForestRegressor(random_state= 2,
                            n_estimators= 100,
                            min_samples_split= 2, 
                            min_samples_leaf= 3, 
                            max_depth= 8, 
                            bootstrap= True)
print('model fitting...')
model.fit(x_train, y_train)
predict_x_test = model.predict(x_test)
print('scoring test...')
score(predict_x_test, y_test)
print('predict with all exist data...')
predict_all_x = model.predict(x)
print(predict_all_x)
print('scoring all data...')
score(predict_all_x, y)
print('predicting...')
predict_newday = prediction(predict_days, model, data_df.iloc[:, 5:9], window_size)
print("create prediction with date...")
last_date = data_df.iloc[-1,'date']
print(last_date)
predict_newday = create_prediction_with_date(predict_newday, predict_days,last_date)
print(predict_newday)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

data_file_name = '2330_data.csv'
window_size = 60
predict_days = 14
split_rate = 0.8
test_days = 0

def getdata_df(file_name):
    data_df = pd.read_csv(file_name)
    return data_df

def define_xy(window_size, train_data):
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
    print("len of train&test data:",len(x))
    #print(x[0])
    #print(type(x[0]))
    #print(y[0])
    #print(type(y[0]))
    return x, y

def split_test_train(test_days, x, y):
    x_test = x[-test_days:]
    x_train = x[:-test_days]
    y_test = y[-test_days:]
    y_train = y[:-test_days]

    return x_train, x_test, y_train, y_test

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
    print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(real, predict)), 4))
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
    #print("***data after prediction:***")
    #print(exist_data.iloc[-(days):,:])
    return exist_data.iloc[-(days):, :]

def create_prediction_with_date(new_data, predict_days, last_date):
    next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
    new_dates = pd.date_range(start=next_date, periods=predict_days, freq='B')
    new_data['date'] = new_dates
    return new_data

# 以下是main
data_df = getdata_df(data_file_name)
print("len of data_df:", len(data_df))
train_days = int((len(data_df)-window_size)*split_rate)
test_days = len(data_df)-window_size-train_days
print("test_days:",test_days)
x, y = define_xy(window_size, data_df)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,  random_state=0)
x_train, x_test, y_train, y_test = split_test_train(test_days, x, y)
print('param tuning...')
# model = hyperparameter_tuning(x_train, y_train) #240個參數tune超久==，得到一次後就自己填:D
model = RandomForestRegressor(random_state= 2,
                            n_estimators= 500,
                            min_samples_split= 2, 
                            min_samples_leaf= 3, 
                            max_depth= 6, 
                            bootstrap= True)
print('model fitting...')
model.fit(x_train, y_train)
print('scoring test...')
predict_x_test = model.predict(x_test)
score(predict_x_test, y_test)
print('scoring train...')
predict_x_train = model.predict(x_train)
score(predict_x_train, y_train)
print('predicting new days...')
predict_newday = prediction(predict_days, model, data_df.iloc[:, 5:9], window_size)
print("create prediction with date...")
last_date = data_df.loc[len(data_df)-1,'date']
print(last_date)
predict_newday = create_prediction_with_date(predict_newday, predict_days, last_date)
print(predict_newday)

plt.figure(figsize=(20,8))
plt.plot(predict_newday['date'], predict_newday['close'], label='new day predict')
plt.legend()
plt.show()
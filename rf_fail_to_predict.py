import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier


# 用FinMind下載資料(有比較齊全ㄉ台股資料)
# 以台積電(編號2330)為例

from FinMind.data import DataLoader

dl = DataLoader()
data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2020-01-01', end_date= '2022-12-31')


# 存成csv
# 把日期設為index
# 使用open, max, min, close, spread, Trading_turnover六個指標進行預測
# 以close的股價作為預測目標
# 用train_test_split切分test&train data
# 並進行標準化(不用這個)

data_df.to_csv("tsmc_stock_from2022.csv")
data1 = pd.read_csv("tsmc_stock_from2022.csv")
data1.set_index("date", inplace=True)
#data1['open'].plot()
#plt.ylabel("open price")
#plt.show()
data1.dropna(inplace = True)
# x = data1.iloc[:-1][['open', 'max', 'min', 'close', 'Trading_turnover']]
x = data1.iloc[:-1,4:8]
print(x)
print(type(x))
y = data1.iloc[:, 4:8]
y = y.shift(periods=-1)
y = y[:-1] #向前移一個
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,  random_state=0)
# scale = StandardScaler()
# x_train = scale.fit_transform(x_train)
# x_test = scale.transform(x_test)


# 用RandomizedSearchCV做hyperparemeter tuning

grid_rf = {
'n_estimators': [20, 50, 100, 500, 1000],  
'max_depth': np.arange(1, 10, 1),  
'min_samples_split': [2, 10, 9], 
'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  
'bootstrap': [True, False], 
'random_state': [1, 2, 30, 42]
}
model = RandomForestRegressor() #?????
rscv = RandomizedSearchCV(estimator=model, param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
rscv_fit = rscv.fit(x_train, y_train)
best_parameters = rscv_fit.best_params_
print(best_parameters)

model = RandomForestRegressor(random_state=best_parameters['random_state'],
                              n_estimators=best_parameters['n_estimators'],
                              min_samples_split=best_parameters['min_samples_split'],
                              min_samples_leaf=best_parameters['min_samples_leaf'],
                              max_depth=best_parameters['max_depth'], 
                              bootstrap=best_parameters['bootstrap']) 
#put param from last step
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(predict)
print(predict.shape)

print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:')
print(round(accuracy, 2), '%.') 


# 嘗試從最近500天預測

x_last500 = data1.iloc[-500:-1,4:8]
y_last500 = data1.iloc[-500:, 4:8]
y_last500 = y_last500.shift(periods = -1)
y_last500 = y_last500[:-1]
print('\n**check x_last500**')
print(x_last500)
print('\n**check y_last500**')
print(y_last500)

# ###這個!!!
# x_last500 = scale.transform(x_last500) 
# ###很重要!!!

last500_predict = model.predict(x_last500)
print('\n**check x_last500 precdict**\n')
print(last500_predict)
print('^_^\n看起來還不錯對八\n^_^')

print('**compare with real value of last 500 days**')
print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_last500, last500_predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_last500, last500_predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_last500, last500_predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_last500, last500_predict), 4))
print(f'Test Score : {model.score(x_last500, y_last500) * 100:.2f}% using Random Tree Regressor.???')
errors = abs(y_last500-last500_predict)
mape = 100 * (errors / y_last500)
accuracy = 100 - np.mean(mape)
print('Accuracy:')
print(round(accuracy, 2), '%.')


#只用昨天預測今天的
n_day = 1
x_last = data1.iloc[-(n_day+1):-1,4:8]
print('\n**check x_last**')
print(x_last)
y_last = data1.iloc[-(n_day+1):, 4:8]
# print(y_last)
y_last = y_last.shift(periods = -1)
y_last = y_last[:-1]
print('\n**check y_last**')
print(y_last)


# ###這個!!!
# x_last = scale.transform(x_last) 
# ###很重要!!!
# print(x_last)



"""
#用5/12預測5/15、5/15預測5/16
x_ = data1.iloc[-1:]
#x_.style.format('{:.2f}')
new_idx = pd.date_range(start='2023-05-15', end='2024-05-15', freq='B') #B:工作日
x_ = x_.reindex(x_.index.union(new_idx))
x_.index = pd.to_datetime(x_.index)
x_.index = x_.index.strftime('%Y-%m-%d')
print(x_)
for i in range(len(x_)-1):
    x_lastDay = x_.iloc[i:i+1, :]
    #print('x_lastDay:')
    #print(x_lastDay)
    # x_lastDay = scale.transform(x_lastDay) 
    nextDay = model.predict(x_lastDay)
    #print('nextDay:')
    #print(nextDay)
    nextDay_series = pd.Series(nextDay[0], index=x_.columns)
    x_.iloc[i+1] = nextDay_series
print(x_)

"""

#我怕是我亂打for所以一個一個做，但還是大失敗

print('************\nLet me try againnnnnnnnn\n**********')
x_ = data1.iloc[-1:, 4:8]
#x_.style.format('{:.2f}')
new_idx = pd.date_range(start='2023-05-15', end='2024-05-15', freq='B') #B:工作日
x_ = x_.reindex(x_.index.union(new_idx))
x_.index = pd.to_datetime(x_.index)
x_.index = x_.index.strftime('%Y-%m-%d')
print(x_)

x_lastDay = x_.iloc[0:0+1, :]
print('x_lastDay:')
print(x_lastDay)
# x_lastDay = scale.transform(x_lastDay) 
nextDay = model.predict(x_lastDay)
print('nextDay:')
print(nextDay)
nextDay_series = pd.Series(nextDay[0], index=x_.columns)
x_.iloc[1] = nextDay_series
print(x_)

x_lastDay = x_.iloc[1:1+1, :]
print('x_lastDay:')
print(x_lastDay)
# x_lastDay = scale.transform(x_lastDay) 
nextDay = model.predict(x_lastDay)
print('nextDay:')
print(nextDay)
nextDay_series = pd.Series(nextDay[0], index=x_.columns)
x_.iloc[2] = nextDay_series
print(x_)

x_lastDay = x_.iloc[2:2+1, :]
print('x_lastDay:')
print(x_lastDay)
# x_lastDay = scale.transform(x_lastDay) 
nextDay = model.predict(x_lastDay)
print('nextDay:')
print(nextDay)
nextDay_series = pd.Series(nextDay[0], index=x_.columns)
x_.iloc[3] = nextDay_series
print(x_)
print('***************\nDAMNNNNN\n***************')

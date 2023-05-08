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
#最後那行是我加ㄉ

# 用FinMind下載資料(有比較齊全ㄉ台股資料)
# 台積電(編號2330)


from FinMind.data import DataLoader

dl = DataLoader()
data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2020-01-01', end_date ='2023-05-07')

# 存成csv
# 把日期設為index
# 使用open, max, min, close, spread, Trading_turnover六個指標進行預測
# 以close的股價作為預測目標
# 用train_test_split切分test&train data

#visualize the data
data_df.to_csv("tsmc_stock_from2020.csv")
data1 = pd.read_csv("tsmc_stock_from2020.csv")
data1.set_index("date", inplace=True)
"""
data1['close'].plot()
plt.ylabel("close price")
plt.show()
"""


data1.dropna(inplace = True)
print(data1.head())

x = data1.iloc[:, 4:10]
print(x)
y = data1.iloc[:, 7]
print(y)
# print("print finish")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=42)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

# 用RandomizedSearchCV做hyperparemeter tuning


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
print(best_parameters)

#然後做預測
# print(best_parameters['random_state'], best_parameters['n_estimators'], best_parameters['max_depth'], best_parameters['min_samples_split'], best_parameters['min_samples_leaf'], best_parameters['bootstrap'])

model = RandomForestRegressor(random_state=best_parameters['random_state'],n_estimators=best_parameters['n_estimators'],min_samples_split=best_parameters['min_samples_split'],min_samples_leaf=best_parameters['min_samples_leaf'],max_depth=best_parameters['max_depth'], bootstrap=best_parameters['bootstrap']) 
#put param from last step
model.fit(x_train, y_train)
predict = model.predict(x_test)
print(predict)
print(predict.shape)

#idk, 一些判斷ㄉ數值

print("Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
print("Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
print("(R^2) Score:", round(metrics.r2_score(y_test, predict), 4))
print(f'Train Score : {model.score(x_train, y_train) * 100:.2f}% and Test Score : {model.score(x_test, y_test) * 100:.2f}% using Random Tree Regressor.')
errors = abs(predict - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.') 


# (為甚麼是529天...?)
# 印出20天後的台積電股價
# 2023/5/9因為今天是5/8


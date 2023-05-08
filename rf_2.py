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
data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2015-01-01')


# 存成csv
# 把日期設為index
# 使用open, max, min, close, spread, Trading_turnover六個指標進行預測
# 以close的股價作為預測目標
# 用train_test_split切分test&train data
# 並進行標準化

data_df.to_csv("tsmc_stock_from2015.csv")
data1 = pd.read_csv("tsmc_stock_from2015.csv")
data1.set_index("date", inplace=True)
#data1['open'].plot()
#plt.ylabel("open price")
#plt.show()
data1.dropna(inplace = True)
x = data1.iloc[:-1, 4:10]
print(x)
y = data1.iloc[:, 7]
y = y.shift(periods=-1)
y = y[:-1] #向前移一個
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26,  random_state=0)
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

print(type(x))


# 用RandomizedSearchCV做hyperparemeter tuning
# (這裡沒懂:O 大玻潰)

grid_rf = {
'n_estimators': [20, 50, 100, 500, 1000],  
'max_depth': np.arange(1, 15, 1),  
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
print('Accuracy:', round(accuracy, 2), '%.') 

print("y_test\t predict")
for i in range(0, 529):
    print(y_test[i],"\t",predict[i])


# 從5/8預測5/9的數據

y_data = data1.iloc[-1:, 4:10]
print(y_data)
y_predict = model.predict(y_data)
print(y_predict)


# ***以下是瑪利亞母虎爛ㄉ***
"""
from datetime import datetime, timedelta
start_date = datetime(2023, 5, 9)
my_list = list(predict)
dates = []
i = 0
while len(dates) < 529:
    date = start_date + timedelta(days=i)
    i+=1
    if date.weekday() < 5:
        dates.append(date)
plt.title('Predict Pirce in next month')
plt.xlabel('Date')
date_strings = [d.strftime('%Y-%m-%d') for d in dates]
plt.xticks(dates, date_strings, rotation=45)
plt.plot(dates[:20], my_list[:20])
plt.show()


# In[173]:


predictions = pd.DataFrame({"Predictions": predict}, index=pd.date_range(start=data1.index[-1], periods=len(predict), freq="D"))
predictions.to_csv("Predicted-price-data.csv")
#colllects future days from predicted values
oneyear_df = pd.DataFrame(predictions[:252])
oneyear_df.to_csv("one-year-predictions.csv")
onemonth_df = pd.DataFrame(predictions[:21])
onemonth_df.to_csv("one-month-predictions.csv")
fivedays_df = pd.DataFrame(predictions[:5])
fivedays_df.to_csv("five-days-predictions.csv")
"""
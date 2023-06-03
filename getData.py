from FinMind.data import DataLoader
import pandas as pd
from pandas import read_csv

dl = DataLoader()
data_df = dl.taiwan_stock_daily(stock_id = '2330', start_date = '2020-01-01', end_date= '2022-12-31')

data_df.to_csv("tsmc_stock_from2022.csv")
data = pd.read_csv("tsmc_stock_from2022.csv")
data.dropna(inplace = True)
data = data.iloc[:-1][[ 'date', 'open', 'max', 'min', 'close']]


print(data.isnull().sum())
print(data.isna().any())

print("Date column data type: ", type(data['date'][0]))
print("Open column data type: ", type(data['open'][0]))
print("Close column data type: ", type(data['close'][0]))
print("High column data type: ", type(data['max'][0]))
print("Low column data type: ", type(data['min'][0]))

data['date'] = pd.to_datetime(data.date)
print(data.head())
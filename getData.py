from FinMind.data import DataLoader
import pandas as pd
from pandas import read_csv


class getData:
    
    
    def __init__(self):
        self.stock_id = 0
        self.startDate = '0000-00-00 00:00:00'
        self.endDate = '0000-00-00 00:00:00' 
        self.data = None
    
    
    def get_stock_data(self, id, start, end):
        self.stock_id = id
        self.startDate = start
        self.endDate = end
        
        dl = DataLoader()
        data_df = dl.taiwan_stock_daily(stock_id = self.stock_id, start_date = self.startDate, end_date= self.endDate)
        data_df.to_csv(str(id) + "_data.csv")
        data = pd.read_csv(str(id) + "_data.csv")
        data.dropna(inplace = True)
        data = data.iloc[:-1][[ 'date', 'open', 'max', 'min', 'close']]
        data['date'] = pd.to_datetime(data.date)
        data.sort_values(by='date', inplace=True)
        self.data = data

        return self.data
    
obj = getData
data = obj.get_stock_data(obj, '2330', '2022-01-01', '2023-05-30')
    

# obj = getData
# data = obj.get_stock_data(obj ,'2330', '2020-01-01', '2022-12-31')



# print(data.isnull().sum())
# print(data.isna().any())

# # print("Date column data type: ", type(data['date'][0]))
# # print("Open column data type: ", type(data['open'][0]))
# # print("Close column data type: ", type(data['close'][0]))
# # print("High column data type: ", type(data['max'][0]))
# # print("Low column data type: ", type(data['min'][0]))
# print(data.head(), data.shape, sep="\n")

# print("Starting date: ",data.iloc[0][0])
# print("Ending date: ", data.iloc[-1][0])
# print("Duration: ", data.iloc[-1][0]-data.iloc[0][0])



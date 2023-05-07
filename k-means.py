import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
import yfinance as yf

start_date = '2020-01-01'
end_date = '2022-01-01'

ticker = 'GOOGL'
data = yf.download(ticker, start_date, end_date)
# print(data.tail())
stock_open = np.array(data['Open']).T 
stock_close = np.array(data['Close']).T
movements = stock_close - stock_open
# print(movements)

plt.figure(figsize = (20,10)) 
plt.subplot(1,2,1) 
plt.title('Company:Google',fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 15)
plt.ylabel('Opening price',fontsize = 15)
plt.plot(data['Open'])
plt.show()

plt.figure(figsize = (20,10)) # Adjusting figure size
plt.title('Company:Google',fontsize = 20)
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 20)
plt.xlabel('Date',fontsize = 20)
plt.ylabel('Price',fontsize = 20)
plt.plot(data.iloc[0:30]['Open'],label = 'Open') # Opening prices of first 30 days are plotted against date
plt.plot(df.iloc[0:30]['Close'],label = 'Close') # Closing prices of first 30 days are plotted against date
plt.legend(loc='upper left', frameon=False,framealpha=1,prop={'size': 22}) # Properties of legend box
plt.show()
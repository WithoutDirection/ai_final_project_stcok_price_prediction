from getData import getData
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn import neighbors
import math
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


data = pd.read_csv('2330_data.csv')

class KNN() :
    def __init__(self,k = 15) :
        self.data = data
        self.K = k
        self.df = None
        self.stock = None
        self.train_data = None
        self.test_data = None
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.neighbor = None

    def extract(self, tagName = 'close'):
        self.df = data[['date',tagName]]
        # print("Shape of close dataframe:", self.df.shape)

    def normalize(self):
        stock = self.df.copy()
        self.stock = stock
        del self.df['date']
        self.df=self.scaler.fit_transform(np.array(self.df).reshape(-1,1))
        
    def split(self, split = 0.8):
        time_step = self.K
        training_size=int(len(self.df)*split)
        test_size=len(self.df)-training_size
        self.train_data,self.test_data=self.df[0:training_size,:],self.df[training_size:len(self.df),:1]
        # print("train_data: ", self.train_data.shape)
        # print("test_data: ", self.test_data.shape)
        
        def create_dataset(dataset):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]    
                dataX.append(a)
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
        
        self.X_train, self.y_train = create_dataset(self.train_data)
        self.X_test, self.y_test = create_dataset(self.test_data)

        # print("X_train: ", self.X_train)
        # print("y_train: ", self.y_train)
        # print("X_test: ", self.X_test)
        # print("y_test", self.y_test)
        
        
    def train(self):
        neighbor = neighbors.KNeighborsRegressor(n_neighbors = self.K)
        neighbor.fit(self.X_train, self.y_train)
        train_predict=neighbor.predict(self.X_train)
        test_predict=neighbor.predict(self.X_test)
        self.neighbor = neighbor
        
        train_predict = train_predict.reshape(-1,1)
        test_predict = test_predict.reshape(-1,1)

        # print("Train data prediction:", train_predict.shape)
        # print("Test data prediction:", test_predict.shape)
        
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        
        def accuracy(test_predict):
            
            original_ytest = self.scaler.inverse_transform(self.y_test.reshape(-1,1))
            
            accuracy = 0
            for i in range(len(test_predict)):
                errors = abs(test_predict[i]-original_ytest[i])
                mape = 100 * (errors / original_ytest[i])
                accuracy += 100 - np.mean(mape)
            accuracy /= len(test_predict)
            
            # Evaluation metrices RMSE and accuracy
            print("Test data RMSE: ", math.sqrt(mean_squared_error(original_ytest,test_predict)))       
            print(f'Test data accuracy: {round(accuracy, 2)}%') 

        
        accuracy (test_predict)
        
        look_back=self.K
        trainPredictPlot = np.empty_like(self.df)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
        # print("Train predicted data: ", trainPredictPlot.shape)

        #shift test predictions for plotting
        testPredictPlot = np.empty_like(self.df)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(self.df)-1, :] = test_predict
        # print("Test predicted data: ", testPredictPlot.shape)


        plotdf = pd.DataFrame({'date': self.stock['date'],
                       'original_close': self.stock['close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

        plt.figure(figsize=(12,5))
        sns.set_style("ticks")
        sns.lineplot(data=plotdf,x="date",y='original_close',color='red', label='original_close')
        sns.lineplot(data=plotdf,x="date",y='train_predicted_close',color='green', label='train_predicted_close')
        sns.lineplot(data=plotdf,x="date",y='test_predicted_close',color='blue', label='test_predicted_close')
        
        sns.despine()
        plt.title("The Stock Price of TSMC",size='x-large',color='blue')
        plt.show()
        
        
    def predict(self, pred_days):
        time_step = self.K
        x_input=self.test_data[len(self.test_data)- time_step:].reshape(1,-1)
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        
        lst_output=[]
        i=0
        
        while(i<pred_days):
    
            if(len(temp_input)>time_step):
        
                x_input=np.array(temp_input[1:])
                #print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
        
                yhat = self.neighbor.predict(x_input)
                #print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat.tolist())
                temp_input=temp_input[1:]
       
                lst_output.extend(yhat.tolist())
                i=i+1
        
            else:
                yhat = self.neighbor.predict(x_input)
        
                temp_input.extend(yhat.tolist())
                lst_output.extend(yhat.tolist())
        
                i=i+1
        
        print("Output of predicted next days: ", len(lst_output))
        last_days=np.arange(1,time_step+1)
        day_pred=np.arange(time_step+1,time_step+pred_days+1)
        
        temp_mat = np.empty((len(last_days)+pred_days+1,1))
        temp_mat[:] = np.nan
        temp_mat = temp_mat.reshape(1,-1).tolist()[0]

        last_original_days_value = temp_mat
        next_predicted_days_value = temp_mat

        last_original_days_value[0:time_step+1] = self.scaler.inverse_transform(self.df[len(self.df)-time_step:]).reshape(1,-1).tolist()[0]
        next_predicted_days_value[time_step+1:] = self.scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]
      
        

        last_date = self.stock['date'].values[-1]
        
        next_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
        new_dates = pd.date_range(start=next_date, periods=pred_days, freq='B')
        future = pd.to_datetime(new_dates)
        
        plotdf = pd.DataFrame({'close_price':next_predicted_days_value[time_step + 1:]})
        plotdf['date'] = future
        # print(plotdf)
        
        plt.figure(figsize=(12,5))
        sns.set_style("ticks")
        sns.lineplot(data=plotdf,x="date",y='close_price', label='future_value')
        
        
        sns.despine()
        plt.title("The Stock Price of TSMC",size='x-large',color='blue')
        plt.show()
       
       
       
KNN = KNN()
KNN.extract()
KNN.normalize()
KNN.split()
KNN.train()
KNN.predict(14)


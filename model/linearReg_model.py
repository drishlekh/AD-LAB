import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class StockPredictorLR:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.time_step = 100
        
    def create_dataset(self, dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)

    def train_model(self):
        try:
            # Load data
            df = pd.read_csv('model/AAPL.csv')
            df['date'] = pd.to_datetime(df['date'])
            df1 = df['close'].values
            
            # Scale data
            df1 = self.scaler.fit_transform(np.array(df1).reshape(-1,1))
            
            # Split data
            training_size = int(len(df1)*0.65)
            train_data = df1[0:training_size,:]
            test_data = df1[training_size:len(df1),:1]
            
            # Create datasets
            X_train, y_train = self.create_dataset(train_data, self.time_step)
            X_test, y_test = self.create_dataset(test_data, self.time_step)
            
            # Train model
            self.model.fit(X_train, y_train)
            
            return True
        except Exception as e:
            print(f"Error in training Linear Regression: {str(e)}")
            return False

    def predict_price(self, target_date):
        try:
            df = pd.read_csv('model/AAPL.csv')
            df['date'] = pd.to_datetime(df['date'])
            
            # Find the row with target date
            target_row = df[df['date'].dt.date == target_date.date()]
            
            if target_row.empty:
                print(f"No data found for date: {target_date}")
                return None
            
            target_idx = target_row.index[0]
            
            if target_idx < self.time_step:
                print("Not enough historical data for prediction")
                return None
            
            # Get the previous 100 days of data
            input_data = df['close'].values[target_idx-self.time_step:target_idx]
            input_data = self.scaler.transform(input_data.reshape(-1, 1))
            input_data = input_data.reshape(1, -1)  # Reshape for Linear Regression
            
            # Make prediction
            prediction = self.model.predict(input_data)
            prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1))
            
            return float(prediction[0][0])
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return None
from xgboost import XGBRegressor

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import math

def XGBPredict(df, type):
    data = df.sort_index(ascending = True, axis = 0)
    new_data = pd.DataFrame(index = range(0, len(df)), columns = ['Date', type])

    if (type == "Close"):
        for i in range(0, len(data)):
            new_data["Date"][i] = data['Date'][i]
            new_data[type][i] = data[type][i]
    elif (type == "Price of change"):
        for i in range(0, len(data)):
            new_data["Date"][i] = data['Date'][i]
            if (i == 0):
                new_data[type][i] = 0
            else:
                new_data[type][i] = (data["Close"][i] - data["Close"][i - 1]) / data["Close"][i - 1] * 100
    elif (type == "RSI"):
        periods = 14
        close_delta = df['Close'].diff()

        up = close_delta.clip(lower = 0)
        down = -1 * close_delta.clip(upper = 0)

        ma_up = up.rolling(periods).mean()
        ma_down = down.rolling(periods).mean()

        rs = ma_up / ma_down
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(0)

        for i in range(0, len(data)):
            new_data["Date"][i] = data['Date'][i]
            new_data[type][i] = rsi[i]
        
    new_data.index = new_data.Date
    new_data.drop("Date", axis = 1, inplace = True)
    dataset = new_data.values

    train_len = math.ceil(len(dataset) * 0.8)
    train = dataset[0:train_len, :]
    valid = dataset[train_len:, :]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])

    X_test = np.array(X_test)

    model = XGBRegressor()
    model.fit(x_train, y_train)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(1, -1))
    predictions = predictions[0]

    train = new_data[:train_len]
    valid = new_data[train_len:]
    valid['Predictions'] = predictions

    return [train, valid]

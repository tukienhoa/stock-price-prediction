from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import math

def RNNPredict(df, type):
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

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    rnn = Sequential()
    rnn.add(SimpleRNN(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    rnn.add(SimpleRNN(units=50))
    rnn.add(Dense(1))

    rnn.compile(loss = 'mean_squared_error', optimizer = 'adam')
    rnn.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2)

    predictions = rnn.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    train = new_data[:train_len]
    valid = new_data[train_len:]
    valid['Predictions'] = predictions

    return [train, valid]
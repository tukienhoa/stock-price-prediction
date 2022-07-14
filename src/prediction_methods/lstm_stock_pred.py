import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

def LSTMPredict():
    df_nse = pd.read_csv("data/NSE-TATA.csv")

    df_nse["Date"] = pd.to_datetime(df_nse.Date,format = "%Y-%m-%d")
    df_nse.index = df_nse['Date']

    data = df_nse.sort_index(ascending = True, axis = 0)
    new_data = pd.DataFrame(index = range(0, len(df_nse)), columns = ['Date', 'Close'])

    for i in range(0, len(data)):
        new_data["Date"][i] = data['Date'][i]
        new_data["Close"][i] = data["Close"][i]
        
    new_data.index = new_data.Date
    new_data.drop("Date", axis = 1, inplace = True)
    dataset = new_data.values

    train = dataset[0:987, :]
    valid = dataset[987:, :]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
        
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    lstm_model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2)

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predictions = lstm_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = predictions

    return [train, valid]
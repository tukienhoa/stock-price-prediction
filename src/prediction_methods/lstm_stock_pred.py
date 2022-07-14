import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

def generateLSTMModel():
    df = pd.read_csv("data/NSE-TATA.csv")
    df.head()

    df["Date"] = pd.to_datetime(df.Date,format = "%Y-%m-%d")
    df.index = df['Date']

    data = df.sort_index(ascending = True, axis = 0)
    new_dataset = pd.DataFrame(index = range(0, len(df)), columns = ['Date','Close'])

    for i in range(0, len(data)):
        new_dataset["Date"][i] = data['Date'][i]
        new_dataset["Close"][i] = data["Close"][i]
        

    new_dataset.index = new_dataset.Date
    new_dataset.drop("Date", axis = 1, inplace = True)
    final_dataset = new_dataset.values

    train_data = final_dataset[0:987,:]
    valid_data = final_dataset[987:,:]

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(final_dataset)

    x_train_data, y_train_data= [], []

    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i, 0])
        y_train_data.append(scaled_data[i, 0])
        
    x_train_data,y_train_data = np.array(x_train_data), np.array(y_train_data)

    x_train_data = np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    inputs_data = new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data = inputs_data.reshape(-1,1)
    inputs_data = scaler.transform(inputs_data)

    lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    lstm_model.fit(x_train_data, y_train_data, epochs = 1, batch_size = 1, verbose = 2)

    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i, 0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_closing_price=lstm_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

    lstm_model.save("models/saved_lstm_model.h5")

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

    model = load_model("models/saved_lstm_model.h5")
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = closing_price

    return [train, valid]
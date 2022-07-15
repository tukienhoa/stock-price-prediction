from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def RNNPredict(x_train, y_train, X_test, new_data, scaler):
    rnn = Sequential()
    rnn.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=50,return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=50,return_sequences=True))
    rnn.add(Dropout(0.2))
    rnn.add(LSTM(units=50))
    rnn.add(Dropout(0.2))
    rnn.add(Dense(units = 1))

    rnn.compile(loss = 'mean_squared_error', optimizer = 'adam')
    rnn.fit(x_train, y_train, epochs = 200, batch_size = 32)

    predictions = rnn.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = predictions

    return [train, valid]
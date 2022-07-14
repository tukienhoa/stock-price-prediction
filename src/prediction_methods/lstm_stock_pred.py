from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def LSTMPredict(x_train, y_train, X_test, new_data, scaler):
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss = 'mean_squared_error', optimizer = 'adam')
    lstm_model.fit(x_train, y_train, epochs = 1, batch_size = 1, verbose = 2)

    predictions = lstm_model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    train = new_data[:987]
    valid = new_data[987:]
    valid['Predictions'] = predictions

    return [train, valid]
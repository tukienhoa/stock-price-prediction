from xgboost import XGBRegressor
import math

def XGBPredict(new_data):
    dataset = new_data.values
    
    train_len = math.ceil(len(dataset) * 0.8)
    train = dataset[0:train_len, :]
    valid = dataset[train_len:, :]

    x_train, y_train = [], []

    for i in range(60, len(train)):
        x_train.append(dataset[i - 60:i, 0])
        y_train.append(dataset[i, 0])

    xgb = XGBRegressor()
    xgb.fit(x_train, y_train)

    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)

    X_test = []
    for i in range(60,inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])

    predictions = xgb.predict(X_test)

    train = new_data[:train_len]
    valid = new_data[train_len:]
    valid['Predictions'] = predictions

    return [train, valid]

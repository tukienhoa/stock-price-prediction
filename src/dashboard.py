import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from prediction_methods.lstm_stock_pred import LSTMPredict
from prediction_methods.xgboost_pred import XGBPredict
from prediction_methods.rnn_stock_pred import RNNPredict

# Run dash
app = dash.Dash()
server = app.server


# Load data
df = pd.read_csv("data/NSE-TATA.csv")

df["Date"] = pd.to_datetime(df.Date,format = "%Y-%m-%d")
df.index = df['Date']

data = df.sort_index(ascending = True, axis = 0)
new_data = pd.DataFrame(index = range(0, len(df)), columns = ['Date', 'Close'])

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

inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# App layout
app.layout = html.Div([
   
    html.H1("Dashboard", style={"textAlign": "center"}),
   
    html.P("Select prediction method: "),
    dcc.Dropdown(["XGBoost", "RNN", "LSTM"], "LSTM", id = "prediction-method-dropdown"),
    html.Div(id="dd-selected-pmethod")
])

# Prediction dropdown callback
@app.callback(Output('dd-selected-pmethod', 'children'),
              Input('prediction-method-dropdown', 'value'))
def update_selected_pmethod(selected_method):
    if (selected_method == "XGBoost"):
        [train, valid] = XGBPredict(new_data)
    elif (selected_method == "RNN"):
        [train, valid] = RNNPredict(x_train, y_train, X_test, new_data, scaler)
    else:
        [train, valid] = LSTMPredict(x_train, y_train, X_test, new_data, scaler)

    return html.Div([
        html.P('Selected prediction method:' + selected_method),
        html.Div([
            html.H2("Actual closing price",style={"textAlign": "center"}),
            dcc.Graph(
                id="Actual Data",
                figure={
                    "data":[
                        go.Scatter(
                            x=train.index,
                            y=valid["Close"],
                            mode='markers'
                        )
                    ],
                    "layout":go.Layout(
                        title='scatter plot',
                        xaxis={'title':'Date'},
                        yaxis={'title':'Closing Rate'}
                    )
                }
            ),
            html.H2("Predicted closing price",style={"textAlign": "center"}),
            dcc.Graph(
                id="Predicted Data",
                figure={
                    "data":[
                        go.Scatter(
                            x = valid.index,
                            y = valid["Predictions"],
                            mode='markers'
                        )
                    ],
                    "layout":go.Layout(
                        title = 'scatter plot',
                        xaxis = {'title':'Date'},
                        yaxis = {'title':'Closing Rate'}
                    )
                }
            )                
        ])                
    ])

# Main
if __name__=='__main__':
    app.run_server(debug=True, use_reloader=False)
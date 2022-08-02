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

train = []
valid = []


# App layout
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    html.Div([
        html.P("Prediction method"),
        dcc.Dropdown(["XGBoost", "RNN", "LSTM"], "LSTM", clearable = False, id = "prediction-method-dropdown"),
        html.P("Prediction type"),
        dcc.Dropdown(["Close", "Price of change"], "Close", clearable = False, id = "prediction-type-dropdown")
    ], id = "user-input"),
    dcc.Loading(
        id="loading-data",
        type="default",
        children=html.Div(id="dd-selected-pmethod")
    )
])

# Prediction dropdown callback
@app.callback(Output('dd-selected-pmethod', 'children'),
              [Input('prediction-method-dropdown', 'value'),
               Input('prediction-type-dropdown', 'value')
              ])
def update_selected_pmethod(selected_method, selected_type):
    if (selected_method == "XGBoost"):
        [train, valid] = XGBPredict(df, selected_type)
    elif (selected_method == "RNN"):
        [train, valid] = RNNPredict(df, selected_type)
    else:
        [train, valid] = LSTMPredict(df, selected_type)
    
    return html.Div([
        html.Div([
            html.H2("Actual " + selected_type, style={"textAlign": "center"}),
            dcc.Graph(
                id="Actual Data",
                figure={
                    "data":[
                        go.Scatter(
                            x=train.index,
                            y=valid[selected_type],
                            mode='markers'
                        )
                    ],
                    "layout":go.Layout(
                        title='scatter plot',
                        xaxis={'title':'Date'},
                        yaxis={'title': selected_type}
                    )
                }
            ),
            html.H2("Predicted " + selected_type, style={"textAlign": "center"}),
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
                        yaxis = {'title': selected_type}
                    )
                }
            )                
        ])                
    ])
   

# Main
if __name__=='__main__':
    app.run_server(debug=True, use_reloader=False)
import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output

import pandas as pd
pd.options.mode.chained_assignment = None

from prediction_methods.lstm_stock_pred import LSTMPredict

app = dash.Dash()
server = app.server

app.layout = html.Div([
   
    html.H1("Dashboard", style={"textAlign": "center"}),
   
    html.P("Select prediction method: "),
    dcc.Dropdown(["XGBoost", "RNN", "LSTM"], "LSTM", id = "prediction-method-dropdown"),
    html.Div(id="dd-selected-pmethod")
])

@app.callback(Output('dd-selected-pmethod', 'children'),
              Input('prediction-method-dropdown', 'value'))
def update_selected_pmethod(selected_method):
    if (selected_method == "XGBoost"):
        return html.Div([
            html.P('Selected prediction method: XGBoost'),
        ])
    elif (selected_method == "RNN"):
        return html.Div([
            html.P('Selected prediction method: RNN'),
        ])
    else:
        [train, valid] = LSTMPredict()
        return html.Div([
            html.P('Selected prediction method: LSTM'),
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
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
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


if __name__=='__main__':
    app.run_server(debug=True, use_reloader=False)
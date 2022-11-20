import dash
import numpy as np
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
from pickle import load
from keras.models import model_from_json
import plotly.graph_objs as go

app = dash.Dash(external_stylesheets=[dbc.themes.SUPERHERO])

# Main Engine KW, max = 39891, min = 1110
min_main_engine_kw = 1000
max_main_engine_kw = 40000

# Deadweight, max = 323183,  min = 6118
min_dwt = 6000
max_dwt = 350000

# GRT, max = 170611, min = 4606
min_grt = 4500
max_grt = 180000

# YB, max = 2020, min= 1993
min_yb = 1990
max_yb = 2022

app.layout = html.Div([
        html.H2('Vessel fuel consumption per Nautical mile prediction (kg/nm)',
            style={'margin-left': '500px', 'margin-bottom': '40px'}),
        html.Div([
          dbc.Label('Vessel\'s Yearbuilt'),
          dcc.Slider(
              min_yb, max_yb, 1,
              marks={i: '{}'.format(i) for i in range(1990, 2023, 2)},
              tooltip={"placement": "bottom", "always_visible": False},
              id='Yearbuilt'
          )
        ],
          style={'width': '75%', 'display': 'inline-block'}),
        html.Hr(),
        html.Div([
          dbc.Label('Vessel\'s  Main Engine KW'),
          dcc.Slider(
              min_main_engine_kw, max_main_engine_kw, 1000,
              marks={i: '{}'.format(i) for i in range(1000, 41000, 2000)},
              tooltip={"placement": "bottom", "always_visible": False},
              # value = '20000',
              id='mainenginekw'
                )
        ],
          style={'width': '75%', 'display': 'inline-block'}),
        html.Hr(),

        html.Div([
          dbc.Label('Vessel\'s  Deadweight'),  # ,{'display':'inline-block','margin-right':20}
          dbc.Input(
              id='deadweight',
              min=min_dwt, max=max_dwt,
              placeholder='Deadweight',
              type='number',  # could be number with step
              # value='100.000',
              style={'width': '25%', 'display': 'flow-root'}
          ),
          dbc.FormText("Min Value: 6.000, Max Value: 350.000"),
        ],
          style={'width': '40%', 'display': 'inline-block'}),
        html.Div([
          dbc.Label('Vessel\'s  Gross Rated Tonnage'),
          # ,{'display':'inline-block','margin-right':20}
          dbc.Input(
              id='grt',
              min=min_grt, max=max_grt,  # step=1000,
              placeholder='Gross Rated Tonnage',
              type='number',  # could be number with step
              # value='100.000',
              # valid=True,
              style={'width': '25%', 'display': 'flow-root'}
          ),
          dbc.FormText("Min Value: 4500, Max Value: 180.000"),
        ],
          style={'width': '40%', 'display': 'inline-block'}),
        html.Hr(),
        html.Div([
          dbc.Button(id='submit_button',
                     n_clicks=0,
                     children='Calculate',
                     style={'margin-bottom': '10px'}
                     )
        ], style={'width': '75%', 'width-bottom': '30px'}),
        # html.Hr(),
        html.Hr(),
        html.Table([
            html.Tr([html.Td(['Linear Model projection of fuel per nautical mile (kg): ']), html.Td(id='linear')]),
            html.Tr([html.Td(['Neural Net projection of fuel per nautical mile (kg): ']), html.Td(id='neural')]),
        ]),
        html.Hr(),
        dcc.Graph(id='graph', style={'width': '48%', 'display': 'inline-block'}),
        dcc.Graph(id='graph2', style={'width': '48%', 'display': 'inline-block'})
        ]
)


@app.callback(
    Output("graph", "figure"),
    Output("graph2", "figure"),
    Output("linear", "children"),
    Output("neural", "children"),
    [Input('submit_button', 'n_clicks')],
    [State("Yearbuilt", "value"),
     State('deadweight', 'value'),
     State('mainenginekw', 'value'),
     State('grt', 'value')]
)
def callback_models(nclicks, yearbuilt, dwt, enginekw, grt):

    consumption = model_linear.predict([[enginekw, dwt, grt, yearbuilt]])
    consumption = np.round(consumption, 2)
    x = [[enginekw, dwt, grt, yearbuilt]]
    x = scaler_input.transform(x)
    consumption2 = model_neural.predict(x)
    consumption2 = scaler_output.inverse_transform(consumption2)
    consumption2 = np.round(float(consumption2), 2)

    data_points_for_graph = []
    vessel_per_year = df[df['Yearbuilt'] == yearbuilt]
    data_points_for_graph.append(go.Scatter(
        x=vessel_per_year['Deadweight'],
        y=vessel_per_year['Annual average Fuel consumption per distance [kg / n mile]'],
        mode='markers',
        opacity=0.7,
        marker={'size': 11},
        name=yearbuilt
    ))

    dwt_min = dwt-(dwt*0.1)
    dwt_max = dwt+(dwt*0.1)
    data_points_for_graph2= []
    vessel_per_dwt = df[(df['Deadweight']>dwt_min) & (df['Deadweight']<dwt_max)]
    data_points_for_graph2.append(go.Scatter(
        x=vessel_per_dwt['Yearbuilt'],
        y=vessel_per_dwt['Annual average Fuel consumption per distance [kg / n mile]'],
        mode='markers',
        opacity=0.7,
        marker={'size': 11},
        name=dwt
    ))

    return ({'data': data_points_for_graph,
            'layout': go.Layout(title='Fuel consumption per nautical mile on same yearbuilt Vessels (kg/nm)',
                                xaxis={'title': 'Deadweight (mt)'},
                                yaxis={'title': 'Fuel per nautical mile (kg)'})},
            {'data': data_points_for_graph2,
            'layout': go.Layout(title='Fuel consumption per nautical mile on same deadweight Vessels +/- 10% (kg/nm)',
                                xaxis={'title': 'Yearbuilt (mt)'},
                                yaxis={'title': 'Fuel per nautical mile (kg)'})},
            consumption,
            consumption2)


if __name__ == '__main__':
    df = pd.read_csv("vessel_information_1.csv")
    model_linear = joblib.load('finalized_lin_model.pkl')
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model_neural = model_from_json(loaded_model_json)
    # load weights into new model
    model_neural.load_weights("model.h5")
    model_neural.compile(loss='mean_squared_error', optimizer='adam')
    scaler_input = load(open('scaler_neural.pkl', 'rb'))
    scaler_output = load(open('scaler_neural_output.pkl', 'rb'))
    app.run_server()

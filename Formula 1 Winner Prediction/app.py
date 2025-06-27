# app.py
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import requests
from src.api_client import get_prediction

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Required for deployment

# Load preprocessed data
data = pd.read_csv('data/processed_data/processed_canadian_gp.csv')

# Layout of the dashboard
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🏎️ Formula 1 Winner Prediction Dashboard", className="text-center mb-4"), width=12)
    ]),
    
    # Dropdown for selecting drivers
    dbc.Row([
        dbc.Col([
            html.Label("Select Driver:"),
            dcc.Dropdown(
                id='driver-dropdown',
                options=[{'label': driver, 'value': driver} for driver in data['DriverNumber'].unique()],
                value=data['DriverNumber'].unique()[0],
                clearable=False
            )
        ], width=6),
        
        dbc.Col([
            html.Label("Predict Winner:"),
            dbc.Button("Predict", id="predict-button", color="primary", className="mt-2")
        ], width=6)
    ]),
    
    # Predicted Winner Section
    dbc.Row([
        dbc.Col(html.Div(id="prediction-output", className="text-center mt-4"), width=12)
    ]),
    
    # Visualization Section
    dbc.Row([
        dbc.Col(dcc.Graph(id="lap-time-chart"), width=12)
    ])
], fluid=True)

# Callback to fetch and display predictions
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    Input("driver-dropdown", "value")
)
def predict_winner(n_clicks, driver):
    if n_clicks is None:
        return "Click 'Predict' to see the winner!"
    
    # Prepare input data for prediction
    input_data = {
        "LapNumber": 50,
        "PitOutTime": 0,
        "Sector1Time": 25.3,
        "Sector2Time": 30.1,
        "Sector3Time": 28.7,
        "Compound_MEDIUM": 1,
        "Compound_SOFT": 0,
        "FastestLap": 1,
        "AvgSectorTime": 28.0
    }
    
    # Call the Flask API for prediction
    prediction = get_prediction(input_data)
    return f"Predicted Winner: {'Yes' if prediction else 'No'}"

# Callback to update the lap time chart
@app.callback(
    Output("lap-time-chart", "figure"),
    Input("driver-dropdown", "value")
)
def update_lap_time_chart(driver):
    filtered_data = data[data['DriverNumber'] == driver]
    fig = px.line(
        filtered_data,
        x="LapNumber",
        y=["Sector1Time", "Sector2Time", "Sector3Time"],
        title=f"Lap Times for Driver {driver}",
        labels={"value": "Time (seconds)", "variable": "Sector"}
    )
    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
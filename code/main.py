import subprocess

# Use subprocess to run the pip install command
try:
    subprocess.check_call(["pip", "install", "dash"])
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pickle
import numpy as np

# Load your machine learning model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Car Price Predictor"),
    html.Label("Engine CC:"),
    dcc.Input(id='engine-input', type='number'),
    html.Label("Mileage kmpl:"),
    dcc.Input(id='mileage-input', type='number'),
    html.Label("Max Power bhp:"),
    dcc.Input(id='power-input', type='number'),
    html.Button(id='calculate-button', n_clicks=0, children='Calculate'),
    html.Div(id='output-price')
])

# Define a callback to calculate and update the result
@app.callback(
    Output('output-price', 'children'),
    Input('calculate-button', 'n_clicks'),
    [dash.dependencies.State('engine-input', 'value'),
     dash.dependencies.State('mileage-input', 'value'),
     dash.dependencies.State('power-input', 'value')]
)
def update_output(n_clicks, engine, mileage, power):
    if n_clicks > 0:
        # Create a data array from user inputs
        data = np.array([[engine, mileage, power]])

        # Use the machine learning model to make a prediction
        predicted_price = model.predict(data)

        # Display the predicted price
        return f"Predicted Car Price: ${predicted_price[0]:.2f}"

if __name__ == '__main__':
    app.run_server(debug=True)

    #app.run_server(debug=True, host='0.0.0.0', port='80') 
    
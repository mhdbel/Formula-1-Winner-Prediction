from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)
model = load('models/random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether a driver will win based on input data.
    """
    data = request.json
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({'winner': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
# src/api_client.py
import requests

def get_prediction(input_data):
    """
    Call the Flask API to get predictions.
    """
    api_url = "http://localhost:5000/predict"
    response = requests.post(api_url, json=input_data)
    if response.status_code == 200:
        return response.json().get("winner", False)
    else:
        raise Exception(f"API Error: {response.status_code}")
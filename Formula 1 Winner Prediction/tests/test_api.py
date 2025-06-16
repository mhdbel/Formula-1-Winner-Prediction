import pytest
import requests

@pytest.fixture
def api_url():
    return "http://localhost:5000/predict"

def test_predict_winner(api_url):
    # Example input data
    input_data = {
        'LapNumber': 50,
        'PitOutTime': 0,
        'Sector1Time': 25.3,
        'Sector2Time': 30.1,
        'Sector3Time': 28.7,
        'Compound_MEDIUM': 1,
        'Compound_SOFT': 0,
        'FastestLap': 1,
        'AvgSectorTime': 28.0
    }
    
    # Send POST request to API
    response = requests.post(api_url, json=input_data)
    
    # Assertions
    assert response.status_code == 200
    assert 'winner' in response.json()
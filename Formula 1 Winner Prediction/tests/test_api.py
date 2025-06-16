import pytest
import pandas as pd
import json
from unittest import mock
from pathlib import Path

# Adjust import path for 'app' and 'preprocess_data' from your 'src' directory
# This often depends on how pytest is invoked and your project structure.
try:
    from src.api import app as flask_app  # flask_app is the Flask app instance
    from src.preprocessing import preprocess_data as actual_preprocess_function
except ImportError:
    print("Attempting fallback import for src.api and src.preprocessing - ensure PYTHONPATH is set if this fails.")
    # Fallback if running tests from a different directory structure
    # This might require specific PYTHONPATH setup or making 'src' an installable package.
    # For example, if tests/ and src/ are siblings:
    # import sys
    # project_root_for_test = Path(__file__).resolve().parent.parent
    # sys.path.insert(0, str(project_root_for_test))
    from ..src.api import app as flask_app
    from ..src.preprocessing import preprocess_data as actual_preprocess_function


# --- Fixtures ---

@pytest.fixture
def client():
    """Create a Flask test client for the API."""
    flask_app.config['TESTING'] = True
    # We need to manage the model and preprocess_data function for tests.
    # Option 1: Let the app load the real model if available and patch preprocess_data.
    # Option 2: Patch both model loading and preprocess_data.

    # For these tests, we'll primarily mock the behavior of the model's predict method
    # and the preprocess_data function to isolate API logic.
    
    with flask_app.test_client() as client:
        yield client

@pytest.fixture
def mock_model_predict():
    """Fixture to mock the model's predict method."""
    # Create a mock model object that an API might load
    mock_model_obj = mock.Mock()
    # Assume it's a scikit-learn model for n_features_in_
    mock_model_obj.n_features_in_ = 3 # Example: model expects 3 features after preprocessing
    
    # Define a default return value for predict
    mock_model_obj.predict.return_value = [1] # Predicts 'winner'
    return mock_model_obj

@pytest.fixture
def mock_preprocess_data():
    """Fixture to mock the preprocess_data function."""
    # This mock will stand in for the actual preprocessing logic.
    # It should return a DataFrame that resembles preprocessed data.
    mock_func = mock.Mock(spec=actual_preprocess_function) # Use spec for better mocking
    
    # Example: preprocess_data returns a DataFrame with 3 features
    sample_processed_df = pd.DataFrame({
        'feature1_proc': [0.5], 'feature2_proc': [1.2], 'feature3_proc': [0] 
    })
    mock_func.return_value = sample_processed_df
    return mock_func

# --- Test Functions ---

def test_health_check_healthy(client):
    """Test the /health endpoint when model and preprocessing are presumably loaded."""
    # This test assumes the global 'model' and 'preprocess_data' in api.py are loaded.
    # If they can be None, this test might need to mock them to be non-None.
    # The api.py was written to allow them to be None and have /health report unhealthy.
    # To test the "healthy" state, we might need to ensure they are patched to be "loaded".
    with mock.patch('src.api.model', new=mock.Mock()), \
         mock.patch('src.api.preprocess_data', new=mock.Mock()):
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json == {"status": "healthy"}

def test_health_check_unhealthy_no_model(client):
    with mock.patch('src.api.model', new=None): # Simulate model not loaded
        response = client.get('/health')
        assert response.status_code == 503
        assert response.json == {"status": "unhealthy", "reason": "Model not loaded"}

def test_health_check_unhealthy_no_preprocessing(client):
    with mock.patch('src.api.preprocess_data', new=None): # Simulate preprocessing not loaded
        response = client.get('/health')
        assert response.status_code == 503
        assert response.json == {"status": "unhealthy", "reason": "Preprocessing module not loaded"}


def test_predict_success(client, mock_model_predict, mock_preprocess_data):
    """Test successful prediction via /predict endpoint."""
    sample_input_data = {"raw_feature1": 10, "raw_feature2": "A"}

    # Patch the globally loaded model and preprocess_data function in the api module
    with mock.patch('src.api.model', new=mock_model_predict), \
         mock.patch('src.api.preprocess_data', new=mock_preprocess_data):
        
        response = client.post('/predict', json=sample_input_data)

    assert response.status_code == 200
    assert response.json == {'winner': True}
    
    # Verify that preprocess_data was called with a DataFrame from input
    # The actual call inside api.py is preprocess_data(input_df.copy())
    # So we check the first argument of the first call to the mock
    assert mock_preprocess_data.call_count == 1
    call_args_df = mock_preprocess_data.call_args[0][0]
    assert isinstance(call_args_df, pd.DataFrame)
    # Crude check if df contains input data; more robust would be comparing df content
    assert call_args_df.iloc[0].get('raw_feature1') == sample_input_data['raw_feature1'] 

    # Verify model's predict was called with the output of preprocess_data
    mock_model_predict.predict.assert_called_once_with(mock_preprocess_data.return_value)


def test_predict_no_input_data(client):
    response = client.post('/predict', json=None) # Or data=None for some clients
    assert response.status_code == 400
    assert "No input data provided" in response.json['error']

def test_predict_bad_json_input(client):
    response = client.post('/predict', data="{bad json", content_type='application/json')
    assert response.status_code == 400 # Flask usually catches malformed JSON
    # The error message might vary based on Flask version and internal handling.
    # It might not be a JSON response if Flask's parser fails early.
    # assert "Failed to decode JSON" in response.get_data(as_text=True) # Example check

def test_predict_preprocessing_fails(client, mock_preprocess_data):
    """Test when preprocess_data itself raises an exception."""
    mock_preprocess_data.side_effect = Exception("Preprocessing boom!")
    sample_input_data = {"feature1": 1}

    with mock.patch('src.api.model', new=mock.Mock()), \
         mock.patch('src.api.preprocess_data', new=mock_preprocess_data):
        response = client.post('/predict', json=sample_input_data)
    
    assert response.status_code == 500 # Internal Server Error
    assert "Error during data preprocessing" in response.json['error']
    assert "Preprocessing boom!" in response.json['error']

def test_predict_feature_mismatch(client, mock_model_predict, mock_preprocess_data):
    """Test when preprocessed data has feature count mismatch with model expectation."""
    # Configure preprocess_data mock to return DF with different number of features
    # than mock_model_predict.n_features_in_ (which is 3)
    mock_preprocess_data.return_value = pd.DataFrame({'feat1': [1], 'feat2': [2]}) # Only 2 features
    sample_input_data = {"feature1": 1}

    with mock.patch('src.api.model', new=mock_model_predict), \
         mock.patch('src.api.preprocess_data', new=mock_preprocess_data):
        response = client.post('/predict', json=sample_input_data)
        
    assert response.status_code == 400
    assert "Feature mismatch after preprocessing" in response.json['error']
    assert "Model expects 3 features, but input has 2 features" in response.json['error']


def test_predict_model_prediction_fails(client, mock_model_predict, mock_preprocess_data):
    """Test when model.predict() raises an exception."""
    mock_model_predict.predict.side_effect = Exception("Model prediction boom!")
    sample_input_data = {"feature1": 1}

    with mock.patch('src.api.model', new=mock_model_predict), \
         mock.patch('src.api.preprocess_data', new=mock_preprocess_data):
        response = client.post('/predict', json=sample_input_data)
        
    assert response.status_code == 500
    assert "Error during prediction" in response.json['error']
    assert "Model prediction boom!" in response.json['error']

def test_predict_model_not_loaded(client):
    """Test /predict when the model is None (failed to load)."""
    with mock.patch('src.api.model', new=None):
        response = client.post('/predict', json={"feature1": 1})
    assert response.status_code == 503 # Service Unavailable
    assert "Model not loaded" in response.json['error']

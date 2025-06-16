import pytest
import pandas as pd
from src.preprocessing import preprocess_data

def test_preprocess_data():
    # Create a dummy DataFrame
    data = pd.DataFrame({
        'Sector1Time': [25.3, None, 26.1],
        'Sector2Time': [30.1, 31.2, None],
        'Sector3Time': [28.7, 29.5, 30.0],
        'Compound': ['MEDIUM', 'SOFT', 'HARD'],
        'IsPersonalBest': [True, False, True]
    })
    
    # Preprocess the data
    processed_data = preprocess_data(data)
    
    # Assertions
    assert not processed_data.isnull().values.any()  # No missing values
    assert 'AvgSectorTime' in processed_data.columns  # New feature added
    assert 'FastestLap' in processed_data.columns     # New feature added
    assert 'Compound_MEDIUM' in processed_data.columns  # One-hot encoding
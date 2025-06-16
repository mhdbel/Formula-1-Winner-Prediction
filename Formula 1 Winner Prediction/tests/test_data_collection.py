import pytest
from src.data_collection import fetch_race_data

def test_fetch_race_data():
    data = fetch_race_data(2023, 'Canadian Grand Prix')
    assert not data.empty
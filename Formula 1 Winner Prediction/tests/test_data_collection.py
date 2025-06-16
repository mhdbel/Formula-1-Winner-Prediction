import pytest
import pandas as pd
from pathlib import Path
from unittest import mock # For mocking fastf1 calls

# Adjust import path as needed
try:
    from src.data_collection import fetch_race_data, save_data, PROJECT_ROOT, FASTF1_CACHE_DIR
except ImportError:
    print("Attempting fallback import for src.data_collection - ensure PYTHONPATH is set if this fails.")
    # This fallback might be needed if your test runner or IDE doesn't see 'src' directly.
    # Consider adding your project root to PYTHONPATH.
    import sys
    # Assuming the script is in tests/ and src/ is a sibling
    # This is a common way to handle imports in tests if src is not installed.
    # However, it's often better to configure PYTHONPATH or use an installable package.
    # For now, this provides a potential fallback.
    # project_root_for_test = Path(__file__).resolve().parent.parent
    # sys.path.insert(0, str(project_root_for_test))
    from ..src.data_collection import fetch_race_data, save_data, PROJECT_ROOT, FASTF1_CACHE_DIR


# --- Mocks and Fixtures ---

@pytest.fixture
def mock_fastf1_session_valid_data():
    """Mocks a fastf1 session object with valid laps and results data."""
    mock_session = mock.Mock()
    
    laps_data = {
        'DriverNumber': ['44', '44', '33', '33'], 'LapNumber': [1, 2, 1, 2],
        'Sector1Time': [pd.Timedelta(seconds=30.1), pd.Timedelta(seconds=30.0), pd.Timedelta(seconds=30.5), pd.Timedelta(seconds=30.4)],
        'Sector2Time': [pd.Timedelta(seconds=35.1), pd.Timedelta(seconds=35.0), pd.Timedelta(seconds=35.5), pd.Timedelta(seconds=35.4)],
        'Sector3Time': [pd.Timedelta(seconds=40.1), pd.Timedelta(seconds=40.0), pd.Timedelta(seconds=40.5), pd.Timedelta(seconds=40.4)],
        'IsPersonalBest': [False, True, False, False]
    }
    mock_session.laps = pd.DataFrame(laps_data)
    
    results_data = {
        'DriverNumber': ['44', '33', '11'], 'Position': [1.0, 2.0, 10.0], 
        'Points': [25.0, 18.0, 1.0], 'TeamName': ['Mercedes', 'Red Bull Racing', 'Red Bull Racing'],
        'Abbreviation': ['HAM', 'VER', 'PER'], 'FullName': ['Lewis Hamilton', 'Max Verstappen', 'Sergio Perez']
    }
    mock_session.results = pd.DataFrame(results_data)
    mock_session.load.return_value = None 
    return mock_session

@pytest.fixture
def mock_fastf1_session_empty_laps():
    """Mocks a fastf1 session with empty laps data."""
    mock_session = mock.Mock()
    mock_session.laps = pd.DataFrame()
    mock_session.results = pd.DataFrame({
        'DriverNumber': ['44', '33'], 'Position': [1.0, 2.0], 'Points': [25, 18],
        'TeamName': ['Mercedes', 'Red Bull Racing'], 'Abbreviation': ['HAM', 'VER'], 'FullName': ['Lewis Hamilton', 'Max Verstappen']
    })
    mock_session.load.return_value = None
    return mock_session

@pytest.fixture
def mock_fastf1_session_no_drivernumber_laps():
    """Mocks a fastf1 session where laps data is missing DriverNumber."""
    mock_session = mock.Mock()
    mock_session.laps = pd.DataFrame({'LapNumber': [1, 2]})
    mock_session.results = pd.DataFrame({
        'DriverNumber': ['44', '33'], 'Position': [1.0, 2.0], 'Points': [25, 18],
        'TeamName': ['Mercedes', 'Red Bull Racing'], 'Abbreviation': ['HAM', 'VER'], 'FullName': ['Lewis Hamilton', 'Max Verstappen']
    })
    mock_session.load.return_value = None
    return mock_session

# --- Test Functions ---

@mock.patch('src.data_collection.fastf1.get_session')
def test_fetch_race_data_success(mock_get_session, mock_fastf1_session_valid_data):
    mock_get_session.return_value = mock_fastf1_session_valid_data
    year, event, session_type = 2023, 'Test Grand Prix', 'R'
    race_df = fetch_race_data(year, event, session_type)
    
    mock_get_session.assert_called_once_with(year, event, session_type)
    mock_fastf1_session_valid_data.load.assert_called_once_with(laps=True, results=True, telemetry=False, weather=False, messages=False)
    
    assert not race_df.empty
    assert 'Win' in race_df.columns
    assert race_df[race_df['DriverNumber'] == '44']['Win'].sum() == 2
    assert race_df[race_df['DriverNumber'] == '33']['Win'].sum() == 0

@mock.patch('src.data_collection.fastf1.get_session')
def test_fetch_race_data_api_error(mock_get_session):
    mock_get_session.side_effect = Exception("FastF1 API Network Error")
    race_df = fetch_race_data(2023, 'Error GP', 'R')
    assert race_df.empty

@mock.patch('src.data_collection.fastf1.get_session')
def test_fetch_race_data_empty_laps(mock_get_session, mock_fastf1_session_empty_laps):
    mock_get_session.return_value = mock_fastf1_session_empty_laps
    race_df = fetch_race_data(2023, 'Empty Laps GP', 'R')
    assert race_df.empty

@mock.patch('src.data_collection.fastf1.get_session')
def test_fetch_race_data_missing_drivernumber_in_laps(mock_get_session, mock_fastf1_session_no_drivernumber_laps):
    mock_get_session.return_value = mock_fastf1_session_no_drivernumber_laps
    race_df = fetch_race_data(2023, 'No DriverNum GP', 'R')
    assert race_df.empty

def test_save_data_creates_file(tmp_path):
    sample_df = pd.DataFrame({'colA': [1, 2], 'colB': ['x', 'y']})
    filename = "test_output.csv"
    
    with mock.patch('src.data_collection.PROJECT_ROOT', tmp_path):
        save_data(sample_df, filename, raw_data_dir_str="test_data/raw_files") 
        
        expected_file = tmp_path / "test_data/raw_files" / filename
        assert expected_file.exists(), f"File should exist at {expected_file}"
        loaded_df = pd.read_csv(expected_file)
        pd.testing.assert_frame_equal(loaded_df, sample_df)

def test_save_data_empty_dataframe(tmp_path, capsys):
    empty_df = pd.DataFrame()
    with mock.patch('src.data_collection.PROJECT_ROOT', tmp_path):
        save_data(empty_df, "empty.csv", raw_data_dir_str="test_data/raw_files")
    
    expected_file = tmp_path / "test_data/raw_files" / "empty.csv"
    assert not expected_file.exists()
    captured = capsys.readouterr()
    assert "Data is empty. Nothing to save." in captured.out

def test_fastf1_cache_directory_created_on_import():
    # This test relies on the import of src.data_collection to have run its course.
    # FASTF1_CACHE_DIR is defined in data_collection using PROJECT_ROOT.
    # We are checking if that directory was actually created.
    # Note: This is testing a side effect of module import, which is a bit unusual
    # but given the code structure, it's a way to check this specific functionality.
    
    # To ensure we're testing the one from the SUT (Software Under Test):
    from src.data_collection import FASTF1_CACHE_DIR as cache_dir_from_module
    
    # We also need to know where PROJECT_ROOT points for the data_collection module
    # to verify against the correct path.
    from src.data_collection import PROJECT_ROOT as project_root_from_module

    # Reconstruct the expected path if FASTF1_CACHE_DIR was not directly imported or is complex
    # expected_cache_dir = project_root_from_module / '.fastf1_cache'
    # assert expected_cache_dir.exists()
    # assert expected_cache_dir.is_dir()
    
    # Simpler: just use the imported FASTF1_CACHE_DIR
    assert cache_dir_from_module.exists(), f"Cache directory {cache_dir_from_module} should exist."
    assert cache_dir_from_module.is_dir(), f"Cache path {cache_dir_from_module} should be a directory."

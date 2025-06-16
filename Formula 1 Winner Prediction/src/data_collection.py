import fastf1
import pandas as pd
from pathlib import Path

# Enable caching for faster data retrieval
fastf1.Cache.enable_cache('cache')

def fetch_race_data(year, event_name):
    """
    Fetch race data for a specific year and event name using fastf1.
    """
    session = fastf1.get_session(year, event_name, 'R')  # 'R' for Race
    session.load()
    results = session.results
    laps = session.laps
    
    # Combine results and lap data
    race_data = laps.merge(results[['DriverNumber', 'Position', 'Points']], on='DriverNumber')
    race_data['Win'] = (race_data['Position'] == 1).astype(int)
    return race_data

def save_data(data, filename):
    """
    Save processed data to a CSV file.
    """
    filepath = Path('data/raw_data') / filename
    data.to_csv(filepath, index=False)

if __name__ == '__main__':
    # Example usage
    canadian_gp_data = fetch_race_data(2023, 'Canadian Grand Prix')
    save_data(canadian_gp_data, 'canadian_gp_2023.csv')
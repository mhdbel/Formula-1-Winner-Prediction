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
    # Ensure 'DriverNumber' is present in both DataFrames for a successful merge
    if 'DriverNumber' not in laps.columns or 'DriverNumber' not in results.columns:
        # Handle cases where 'DriverNumber' might be missing or named differently
        # This might involve logging an error or raising an exception
        print("Error: 'DriverNumber' column missing in lap or results data.")
        # Depending on desired robustness, you might return an empty DataFrame or partial data
        # For now, let's assume it's present or the original behavior is acceptable if it errors out
        pass # Allow original merge to proceed and potentially fail, or handle more gracefully

    # A more robust merge might involve checking if results or laps are empty
    if laps.empty or results.empty:
        print("Warning: Laps or results data is empty. Cannot perform full merge.")
        # Return an empty or partially processed DataFrame as appropriate
        # For this example, we'll proceed but this is a point of fragility.
        # A truly robust solution might create an empty DataFrame with expected columns.
        return pd.DataFrame() # Or handle as per application requirements

    race_data = laps.merge(results[['DriverNumber', 'Position', 'Points']], on='DriverNumber', how='left')
    race_data['Win'] = (race_data['Position'] == 1).astype(int)
    return race_data

def save_data(data, filename):
    """
    Save processed data to a CSV file.
    """
    filepath = Path('data/raw_data') / filename
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        data.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")
        # Potentially re-raise the exception or handle as needed

if __name__ == '__main__':
    # Example usage
    try:
        # It's good practice to ensure fastf1 cache is enabled here too if running standalone
        # fastf1.Cache.enable_cache('cache') # Already enabled globally, but good for explicitness
        
        print("Fetching data for Canadian Grand Prix 2023...")
        canadian_gp_data = fetch_race_data(2023, 'Canadian Grand Prix')
        
        if not canadian_gp_data.empty:
            print("Saving data...")
            save_data(canadian_gp_data, 'canadian_gp_2023.csv')
        else:
            print("No data fetched, skipping save.")
            
    except Exception as e:
        print(f"An error occurred in the main execution block: {e}")

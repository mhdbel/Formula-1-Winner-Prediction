import fastf1
import pandas as pd
from pathlib import Path
from src import config
from src.utils import logger # Import logger

# Define project root and cache directory (these are specific to data_collection's needs)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FASTF1_CACHE_DIR = PROJECT_ROOT / '.fastf1_cache'

# Create cache directory if it doesn't exist
FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Enable caching for faster data retrieval
fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

def fetch_race_data(year, event_name):
    """
    Fetch race data for a specific year and event name using fastf1.
    """
    try:
        session = fastf1.get_session(year, event_name, 'R')  # 'R' for Race
        session.load() # Consider selective loading: laps=True, results=True, telemetry=False, etc.
    except Exception as e:
        logger.error(f"Failed to fetch data session for year {year}, event {event_name}: {e}", exc_info=True)
        return pd.DataFrame()

    results = session.results
    laps = session.laps

    if 'DriverNumber' not in laps.columns or 'DriverNumber' not in results.columns:
        logger.error("DriverNumber column missing in lap or results data.")
        # Depending on desired robustness, could raise error or return specific empty DF
        return pd.DataFrame()

    if laps.empty or results.empty:
        logger.warning("Laps or results data is empty. Cannot perform full merge.")
        return pd.DataFrame()

    try:
        race_data = laps.merge(results[['DriverNumber', 'Position', 'Points']], on='DriverNumber', how='left')
        race_data['Win'] = (race_data['Position'] == 1).astype(int)
    except Exception as e:
        logger.error(f"Error merging laps and results data for {year} {event_name}: {e}", exc_info=True)
        return pd.DataFrame()

    return race_data

def save_data(data, filename):
    """
    Save processed data to a CSV file in the raw_data directory.
    """
    filepath = config.RAW_DATA_DIR / filename
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}", exc_info=True)

if __name__ == '__main__':
    try:
        logger.info("Fetching data for Canadian Grand Prix 2023...")
        canadian_gp_data = fetch_race_data(2023, 'Canadian Grand Prix')

        if not canadian_gp_data.empty:
            logger.info("Saving data...")
            save_data(canadian_gp_data, config.DEFAULT_RAW_FILENAME)
        else:
            logger.info("No data fetched, skipping save.")

    except Exception as e:
        logger.error(f"An error occurred in the data_collection main execution block: {e}", exc_info=True)

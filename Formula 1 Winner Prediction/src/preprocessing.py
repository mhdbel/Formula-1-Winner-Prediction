import pandas as pd
from pathlib import Path
from src import config
from src.utils import logger # Import logger

def preprocess_data(data):
    """
    Preprocess raw data for modeling.
    Assumes 'data' is a pandas DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        logger.error("Input data is not a pandas DataFrame.")
        return pd.DataFrame()

    cols_to_drop = ['Time', 'Driver', 'Team', 'LapTime']
    existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    data = data.drop(columns=existing_cols_to_drop, errors='ignore')

    sector_time_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in sector_time_cols:
        if col in data.columns:
            if data[col].isnull().any():
                data[col].fillna(data[col].mean(), inplace=True)
        else:
            logger.warning(f"Column {col} not found for missing value imputation.")

    if all(col in data.columns for col in sector_time_cols):
        data['AvgSectorTime'] = data[sector_time_cols].mean(axis=1)
    else:
        logger.warning("Not all sector time columns present for AvgSectorTime calculation.")
        data['AvgSectorTime'] = pd.NA

    if 'IsPersonalBest' in data.columns:
        data['FastestLap'] = (data['IsPersonalBest']).astype(int)
    else:
        logger.warning("Column 'IsPersonalBest' not found for 'FastestLap' feature.")
        data['FastestLap'] = 0

    # Handle 'Compound' column for one-hot encoding
    if 'Compound' in data.columns:
        # Ensure 'Compound' is treated as categorical to handle potential mixed types if read from CSV
        data['Compound'] = data['Compound'].astype('category')
        # Explicitly define categories to ensure consistent dummy variable creation
        # This helps if not all compounds are present in a given batch of data.
        # Order matters if drop_first=True is used and categories are not sorted by get_dummies.
        # Alphabetical sort: HARD, MEDIUM, SOFT. If HARD is dropped, Compound_MEDIUM, Compound_SOFT created.
        # If we want to ensure a specific one is dropped (e.g. 'HARD'), we can set categories.
        # However, pandas get_dummies sorts lexical by default for drop_first if not a factor.
        # To be explicit about which columns are created:
        # compounds = ['HARD', 'MEDIUM', 'SOFT'] # Example, adjust to actual data
        # data['Compound'] = pd.Categorical(data['Compound'], categories=compounds, ordered=False)
        data = pd.get_dummies(data, columns=['Compound'], prefix='Compound', drop_first=True)
    else:
        logger.warning("Column 'Compound' not found. Creating dummy columns ('Compound_MEDIUM', 'Compound_SOFT') with all zeros.")
        # Assuming 'HARD' is the baseline category that would be dropped by drop_first=True.
        # If 'Compound' column was present and pd.get_dummies(..., drop_first=True) was used,
        # and if compounds were 'HARD', 'MEDIUM', 'SOFT', pandas would sort them,
        # drop 'Compound_HARD' (first alphabetically), and create 'Compound_MEDIUM', 'Compound_SOFT'.
        data['Compound_MEDIUM'] = 0
        data['Compound_SOFT'] = 0
        # If other compounds are possible (e.g. 'WET', 'INTERMEDIATE'), they would also need to be handled.
        # For this example, sticking to SOFT, MEDIUM, HARD.

    return data

def save_processed_data(data, filename):
    """
    Save preprocessed data to a CSV file in the processed_data directory.
    """
    filepath = config.PROCESSED_DATA_DIR / filename
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving processed data to {filepath}: {e}", exc_info=True)

if __name__ == '__main__':
    raw_data_dir = config.RAW_DATA_DIR
    raw_data_filename = config.DEFAULT_RAW_FILENAME
    raw_data_path = raw_data_dir / raw_data_filename
    processed_data_filename = config.DEFAULT_PROCESSED_FILENAME

    logger.info(f"Attempting to preprocess data from: {raw_data_path}")

    if raw_data_path.exists():
        try:
            raw_data_df = pd.read_csv(raw_data_path)
            if not raw_data_df.empty:
                logger.info("Raw data loaded successfully. Starting preprocessing...")
                processed_df = preprocess_data(raw_data_df.copy())

                if not processed_df.empty:
                    logger.info("Preprocessing complete. Saving processed data...")
                    save_processed_data(processed_df, processed_data_filename)
                else:
                    logger.warning("Preprocessing resulted in an empty DataFrame. Not saving.")
            else:
                logger.warning(f"Raw data file {raw_data_path} is empty.")
        except pd.errors.EmptyDataError:
            logger.error(f"The file {raw_data_path} is empty or not a valid CSV.", exc_info=True)
        except Exception as e:
            logger.error(f"An error occurred during the preprocessing pipeline: {e}", exc_info=True)
    else:
        logger.error(f"Raw data file not found at {raw_data_path}.")
        logger.info("Please ensure 'data_collection.py' has run successfully or that a raw data file exists at this location.")

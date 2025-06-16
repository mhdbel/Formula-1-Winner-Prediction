import pandas as pd
from pathlib import Path

def preprocess_data(data):
    """
    Preprocess raw data for modeling.
    Assumes 'data' is a pandas DataFrame.
    """
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        print("Error: Input data is not a pandas DataFrame.")
        return pd.DataFrame() # Return empty DataFrame or raise error

    # Drop unnecessary columns
    # Add defensive check for column existence before dropping
    cols_to_drop = ['Time', 'Driver', 'Team', 'LapTime']
    existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    data = data.drop(columns=existing_cols_to_drop, errors='ignore') # errors='ignore' is good, but explicit check is better
    
    # Handle missing values for sector times
    # Check if sector time columns exist before trying to fill
    sector_time_cols = ['Sector1Time', 'Sector2Time', 'Sector3Time']
    for col in sector_time_cols:
        if col in data.columns:
            if data[col].isnull().any(): # Only fill if there are NaNs
                data[col].fillna(data[col].mean(), inplace=True)
        else:
            print(f"Warning: Column {col} not found for missing value imputation.")
            # Consider creating these columns with NaNs or a default if they are essential downstream
            # For now, we'll assume they might be missing and the model can handle it or it's an issue.

    # Feature engineering
    # Ensure sector time columns are present for AvgSectorTime
    if all(col in data.columns for col in sector_time_cols):
        data['AvgSectorTime'] = data[sector_time_cols].mean(axis=1)
    else:
        print("Warning: Not all sector time columns present for AvgSectorTime calculation.")
        data['AvgSectorTime'] = pd.NA # Or 0 or mean, depending on downstream needs

    # Ensure IsPersonalBest column exists
    if 'IsPersonalBest' in data.columns:
        data['FastestLap'] = (data['IsPersonalBest']).astype(int)
    else:
        print("Warning: Column 'IsPersonalBest' not found for 'FastestLap' feature.")
        data['FastestLap'] = 0 # Default value

    # Encode categorical variables
    # Ensure Compound column exists
    if 'Compound' in data.columns:
        data = pd.get_dummies(data, columns=['Compound'], prefix='Compound', drop_first=True)
    else:
        print("Warning: Column 'Compound' not found for one-hot encoding.")
        # This will result in missing dummy columns that the model might expect.
        # A robust solution would be to add known compound columns with 0 if 'Compound' is missing.
    
    return data

def save_processed_data(data, filename):
    """
    Save preprocessed data to a CSV file.
    """
    filepath = Path('data/processed_data') / filename
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        data.to_csv(filepath, index=False)
        print(f"Processed data saved to {filepath}")
    except Exception as e:
        print(f"Error saving processed data to {filepath}: {e}")

if __name__ == '__main__':
    # Example usage:
    # Create a dummy raw data file for the example if it doesn't exist,
    # or point to an existing one.
    
    # Define path for the raw data CSV
    raw_data_dir = Path('data/raw_data')
    raw_data_filename = 'canadian_gp_2023.csv' # Example filename
    raw_data_path = raw_data_dir / raw_data_filename

    processed_data_filename = 'processed_canadian_gp.csv' # Example output

    print(f"Attempting to preprocess data from: {raw_data_path}")

    if raw_data_path.exists():
        try:
            raw_data_df = pd.read_csv(raw_data_path)
            if not raw_data_df.empty:
                print("Raw data loaded successfully. Starting preprocessing...")
                processed_df = preprocess_data(raw_data_df.copy()) # Use .copy() as preprocess_data might modify
                
                if not processed_df.empty:
                    print("Preprocessing complete. Saving processed data...")
                    save_processed_data(processed_df, processed_data_filename)
                else:
                    print("Preprocessing resulted in an empty DataFrame. Not saving.")
            else:
                print(f"Raw data file {raw_data_path} is empty.")
        except pd.errors.EmptyDataError:
            print(f"Error: The file {raw_data_path} is empty or not a valid CSV.")
        except Exception as e:
            print(f"An error occurred during the preprocessing pipeline: {e}")
    else:
        print(f"Error: Raw data file not found at {raw_data_path}.")
        print("Please ensure 'data_collection.py' has run successfully or that a raw data file exists at this location.")

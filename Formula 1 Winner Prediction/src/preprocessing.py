import pandas as pd

def preprocess_data(data):
    """
    Preprocess raw data for modeling.
    """
    # Drop unnecessary columns
    data = data.drop(columns=['Time', 'Driver', 'Team', 'LapTime'], errors='ignore')
    
    # Handle missing values
    data.fillna({'Sector1Time': data['Sector1Time'].mean(),
                 'Sector2Time': data['Sector2Time'].mean(),
                 'Sector3Time': data['Sector3Time'].mean()}, inplace=True)
    
    # Feature engineering
    data['AvgSectorTime'] = data[['Sector1Time', 'Sector2Time', 'Sector3Time']].mean(axis=1)
    data['FastestLap'] = (data['IsPersonalBest']).astype(int)
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=['Compound'], drop_first=True)
    
    return data

def save_processed_data(data, filename):
    """
    Save preprocessed data to a CSV file.
    """
    filepath = Path('data/processed_data') / filename
    data.to_csv(filepath, index=False)

if __name__ == '__main__':
    # Example usage
    raw_data = pd.read_csv('data/raw_data/canadian_gp_2023.csv')
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data, 'processed_canadian_gp.csv')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load # Added load for completeness in example
from pathlib import Path

def train_model(X_train, y_train, model_params=None):
    """
    Train a Random Forest classifier.
    model_params: dict, optional, parameters for RandomForestClassifier
    """
    if model_params is None:
        model_params = {'n_estimators': 100, 'random_state': 42} # Default parameters
    
    model = RandomForestClassifier(**model_params)
    try:
        model.fit(X_train, y_train)
        print("Model training complete.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return None # Return None if training fails
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    Returns a dictionary with evaluation metrics.
    """
    if model is None:
        print("Error: Model is None, cannot evaluate.")
        return None

    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True) # Get report as dict
        
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred)) # Print human-readable report
        
        return {"accuracy": accuracy, "classification_report": report}
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

def save_model(model, filepath_str):
    """
    Save the trained model to a file.
    filepath_str: string, path to save the model.
    """
    if model is None:
        print("Error: Model is None, cannot save.")
        return

    filepath = Path(filepath_str)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        dump(model, filepath)
        print(f"Model saved to {filepath}")
    except Exception as e:
        print(f"Error saving model to {filepath}: {e}")

if __name__ == '__main__':
    processed_data_dir = Path('data/processed_data')
    processed_data_filename = 'processed_canadian_gp.csv' # Example filename from preprocessing step
    processed_data_path = processed_data_dir / processed_data_filename

    model_output_dir = Path('models')
    model_filename = 'random_forest_model.joblib' # Using .joblib extension
    model_output_path = model_output_dir / model_filename
    
    print(f"Attempting to load processed data from: {processed_data_path}")

    if processed_data_path.exists():
        try:
            data_df = pd.read_csv(processed_data_path)
            if data_df.empty:
                raise ValueError("Processed data file is empty.")

            # Define features (X) and target (y)
            # This assumes 'Win' is the target and other specific columns might be dropped or are not features
            # Ensure these columns exist before dropping
            cols_to_drop_for_X = ['Position', 'Points', 'Win'] # Example, adjust as per actual feature set
            existing_cols_to_drop = [col for col in cols_to_drop_for_X if col in data_df.columns]
            
            if 'Win' not in data_df.columns:
                raise ValueError("Target column 'Win' not found in processed data.")

            X = data_df.drop(columns=existing_cols_to_drop)
            y = data_df['Win']
            
            # Ensure X is not empty after drops
            if X.empty:
                raise ValueError("Feature set X is empty after dropping columns.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)
            
            print("Training model...")
            # Example of passing model parameters
            custom_model_params = {'n_estimators': 150, 'random_state': 42, 'max_depth': 10}
            trained_model = train_model(X_train, y_train, model_params=custom_model_params)
            
            if trained_model:
                print("Evaluating model...")
                evaluation_results = evaluate_model(trained_model, X_test, y_test)
                if evaluation_results:
                    print("Saving model...")
                    save_model(trained_model, model_output_path)
                    
                    # Example of loading the model back (optional)
                    # print("Attempting to load the saved model...")
                    # loaded_model = load(model_output_path)
                    # print("Model loaded successfully. Can be used for predictions.")
            else:
                print("Model training failed. Skipping evaluation and saving.")

        except FileNotFoundError:
            print(f"Error: Processed data file not found at {processed_data_path}.")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An error occurred in the modeling pipeline: {e}")
    else:
        print(f"Error: Processed data file not found at {processed_data_path}.")
        print("Please ensure 'preprocessing.py' has run successfully or that a processed data file exists.")

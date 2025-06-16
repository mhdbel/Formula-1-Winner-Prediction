import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
from pathlib import Path
from src import config
from src.utils import logger # Import logger

def train_model(X_train, y_train, model_params=None):
    """
    Train a Random Forest classifier.
    model_params: dict, optional, parameters for RandomForestClassifier
    """
    if model_params is None:
        model_params = config.DEFAULT_MODEL_PARAMS.copy()

    model = RandomForestClassifier(**model_params)
    try:
        model.fit(X_train, y_train)
        logger.info("Model training complete.")
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return None
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    Returns a dictionary with evaluation metrics.
    """
    if model is None:
        logger.error("Model is None, cannot evaluate.")
        return None

    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report_str = classification_report(y_test, y_pred) # Get string report for logging
        report_dict = classification_report(y_test, y_pred, output_dict=True) # Get dict for return

        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{report_str}")

        return {"accuracy": accuracy, "classification_report": report_dict}
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        return None

def save_model(model, filepath):
    """
    Save the trained model to a file.
    filepath: Path object, path to save the model.
    """
    if model is None:
        logger.error("Error: Model is None, cannot save.")
        return

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}", exc_info=True)

if __name__ == '__main__':
    processed_data_dir = config.PROCESSED_DATA_DIR
    processed_data_filename = config.DEFAULT_PROCESSED_FILENAME
    processed_data_path = processed_data_dir / processed_data_filename
    model_output_path = config.MODEL_PATH

    logger.info(f"Attempting to load processed data from: {processed_data_path}")

    if processed_data_path.exists():
        try:
            data_df = pd.read_csv(processed_data_path)
            if data_df.empty:
                # Log and raise to be caught by the general Exception block, or handle specifically
                logger.error("Processed data file is empty.")
                raise ValueError("Processed data file is empty.")

            cols_to_drop_for_X = ['Position', 'Points', 'Win']
            existing_cols_to_drop = [col for col in cols_to_drop_for_X if col in data_df.columns]

            if 'Win' not in data_df.columns:
                logger.error("Target column 'Win' not found in processed data.")
                raise ValueError("Target column 'Win' not found in processed data.")

            X = data_df.drop(columns=existing_cols_to_drop)
            y = data_df['Win']

            if X.empty:
                logger.error("Feature set X is empty after dropping columns.")
                raise ValueError("Feature set X is empty after dropping columns.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None)

            logger.info("Training model...")
            custom_model_params = config.EXAMPLE_TRAINING_PARAMS.copy()
            trained_model = train_model(X_train, y_train, model_params=custom_model_params)

            if trained_model:
                logger.info("Evaluating model...")
                evaluation_results = evaluate_model(trained_model, X_test, y_test)
                if evaluation_results:
                    logger.info("Saving model...")
                    save_model(trained_model, model_output_path)

                    # logger.info("Attempting to load the saved model...")
                    # loaded_model = load(model_output_path)
                    # logger.info("Model loaded successfully. Can be used for predictions.")
            else:
                logger.warning("Model training failed. Skipping evaluation and saving.")

        except FileNotFoundError:
            logger.error(f"Processed data file not found at {processed_data_path}.", exc_info=True)
        except ValueError as ve:
            logger.error(f"ValueError in modeling pipeline: {ve}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred in the modeling pipeline: {e}", exc_info=True)
    else:
        logger.error(f"Processed data file not found at {processed_data_path}.")
        logger.info("Please ensure 'preprocessing.py' has run successfully or that a processed data file exists.")

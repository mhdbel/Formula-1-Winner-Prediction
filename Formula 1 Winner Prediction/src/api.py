from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from pathlib import Path
from src import config
from .preprocessing import preprocess_data
from src.utils import logger # Import logger

# --- Application Setup ---
app = Flask(__name__)

# --- Load Model ---
MODEL_PATH = config.MODEL_PATH

model = None
if not MODEL_PATH.exists():
    logger.critical(f"Model file not found at {MODEL_PATH}. API will not be able to make predictions.")
else:
    try:
        model = load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.critical(f"Failed to load model from {MODEL_PATH}. Error: {e}. API will not be able to make predictions.", exc_info=True)

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether a driver will win based on input data.
    Input data must be a JSON object.
    """
    if model is None: # This check is important
        logger.error("Model is not loaded, cannot make predictions.")
        return jsonify({"error": "Model not loaded. API cannot make predictions."}), 503

    data = request.json
    if not data:
        logger.warning("No input data provided for /predict endpoint.")
        return jsonify({"error": "No input data provided."}), 400

    try:
        input_df = pd.DataFrame([data] if isinstance(data, dict) else data)
    except Exception as e:
        logger.error(f"Failed to create DataFrame from input: {e}", exc_info=True)
        return jsonify({"error": f"Failed to create DataFrame from input: {e}. Input should be a JSON object or a list of JSON objects."}), 400

    if input_df.empty:
        logger.warning("Input data resulted in an empty DataFrame.")
        return jsonify({"error": "Input data resulted in an empty DataFrame."}), 400

    try:
        processed_input_df = preprocess_data(input_df.copy())
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}", exc_info=True)
        return jsonify({"error": f"Error during data preprocessing. Check server logs for details."}), 500

    if processed_input_df.empty:
        logger.warning("Preprocessing resulted in empty data. Cannot predict.")
        return jsonify({"error": "Preprocessing resulted in empty data. Cannot predict."}), 400

    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
        actual_features = processed_input_df.shape[1]
        if actual_features != expected_features:
            error_msg = (
                f"Feature mismatch after preprocessing: "
                f"Model expects {expected_features} features, but input has {actual_features} features. "
                f"Ensure input data schema matches the training data schema after preprocessing."
            )
            logger.error(error_msg)
            logger.debug(f"Processed input columns: {processed_input_df.columns.tolist()}")
            return jsonify({"error": error_msg}), 400
    else:
        logger.warning("Cannot automatically verify number of input features for the loaded model type.")

    try:
        predictions = model.predict(processed_input_df)
        if isinstance(data, dict):
            result = {'winner': bool(predictions[0])}
        else:
            result = [{'winner': bool(p)} for p in predictions]
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": f"Error during prediction. Check server logs."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 503
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    logger.info(f"Starting Flask API server on {config.API_HOST}:{config.API_PORT} (Debug: {config.API_DEBUG_MODE})")

    if model is None: # Reminder if model loading failed at startup
        logger.warning("Reminder: Model was not loaded at startup. /predict endpoint will not work correctly.")

    app.run(host=config.API_HOST, port=config.API_PORT, debug=config.API_DEBUG_MODE)

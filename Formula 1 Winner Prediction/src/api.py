rom flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from pathlib import Path

# Attempt to import the preprocessing function
try:
    from src.preprocessing import preprocess_data
except ImportError:
    # This is a fallback if the src structure is not directly importable,
    # e.g. if api.py is run from within src/ directory.
    # A better solution is to ensure the Python path is set up correctly
    # or the project is installed as a package.
    try:
        from preprocessing import preprocess_data
    except ImportError:
        preprocess_data = None
        print("CRITICAL ERROR: Could not import preprocess_data function.")
        print("Ensure api.py is run from the project root or PYTHONPATH is set correctly.")


# --- Application Setup ---
app = Flask(__name__)

# --- Load Model ---
# Construct the model path relative to this file's location
# Assumes api.py is in 'src/', models are in 'models/' at the project root.
MODEL_NAME = 'random_forest_model.joblib' # Ensure this matches the saved model name
try:
    API_FILE_PATH = Path(__file__).resolve()
    PROJECT_ROOT = API_FILE_PATH.parent.parent # Moves up two levels (src -> project root)
    MODEL_PATH = PROJECT_ROOT / 'models' / MODEL_NAME
except NameError: # __file__ is not defined (e.g. if running in some interactive environments)
    print("Warning: __file__ not defined. Assuming model path relative to current working directory.")
    MODEL_PATH = Path('models') / MODEL_NAME


model = None
if not MODEL_PATH.exists():
    print(f"CRITICAL ERROR: Model file not found at {MODEL_PATH}")
    print("API will not be able to make predictions.")
else:
    try:
        model = load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model from {MODEL_PATH}. Error: {e}")
        print("API will not be able to make predictions.")

if preprocess_data is None and model is not None:
    print("CRITICAL WARNING: preprocess_data function not loaded. API predictions will likely fail or be incorrect.")

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether a driver will win based on input data.
    Input data must be a JSON object.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. API cannot make predictions."}), 503 # Service Unavailable

    if preprocess_data is None:
        return jsonify({"error": "Preprocessing module not loaded. API cannot process data."}), 503

    data = request.json
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    try:
        # Ensure data is a list of dicts for DataFrame creation if multiple samples,
        # or a single dict for a single sample. pd.DataFrame expects a list of records.
        input_df = pd.DataFrame([data] if isinstance(data, dict) else data)
    except Exception as e:
        return jsonify({"error": f"Failed to create DataFrame from input: {e}. Input should be a JSON object or a list of JSON objects."}), 400

    if input_df.empty:
        return jsonify({"error": "Input data resulted in an empty DataFrame."}), 400

    try:
        # Preprocess the input data
        # Use .copy() if preprocess_data modifies inplace, though it shouldn't based on its current design.
        processed_input_df = preprocess_data(input_df.copy())
    except Exception as e:
        print(f"Error during data preprocessing: {e}") # Log this for server-side debugging
        return jsonify({"error": f"Error during data preprocessing: {e}. Check server logs for details."}), 500

    if processed_input_df.empty:
        return jsonify({"error": "Preprocessing resulted in empty data. Cannot predict."}), 400

    # Feature consistency check (important!)
    # This check assumes 'model' is a scikit-learn model with n_features_in_
    if hasattr(model, 'n_features_in_'):
        expected_features = model.n_features_in_
        actual_features = processed_input_df.shape[1]
        if actual_features != expected_features:
            # Before failing, check if model was trained with feature names
            # and if processed_input_df has those names.
            # This part can get complex if features were selected/reordered after initial preprocessing.
            # For now, a simple count check.
            error_msg = (
                f"Feature mismatch after preprocessing: "
                f"Model expects {expected_features} features, but input has {actual_features} features. "
                f"Ensure input data schema matches the training data schema after preprocessing."
            )
            print(error_msg) # Log this
            # print(f"Model expected columns (if available): {model.feature_names_in_}")
            print(f"Processed input columns: {processed_input_df.columns.tolist()}")
            return jsonify({"error": error_msg}), 400
    else:
        # If the model object doesn't have n_features_in_ (e.g. older scikit-learn, custom model)
        # This check is harder. We could log a warning.
        print("Warning: Cannot automatically verify number of input features for the loaded model type.")

    try:
        predictions = model.predict(processed_input_df)
        # Assuming prediction is an array, convert to list of bools
        # If input was a single dict, predictions[0] is fine. If multiple, return list.
        if isinstance(data, dict): # Single prediction
            result = {'winner': bool(predictions[0])}
        else: # Multiple predictions
            result = [{'winner': bool(p)} for p in predictions]
        return jsonify(result)
    except Exception as e:
        print(f"Error during prediction: {e}") # Log this
        return jsonify({"error": f"Error during prediction: {e}. Check server logs."}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    if model is None:
        return jsonify({"status": "unhealthy", "reason": "Model not loaded"}), 503
    if preprocess_data is None:
        return jsonify({"status": "unhealthy", "reason": "Preprocessing module not loaded"}), 503
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    # Make host and port configurable, e.g., via environment variables or a config file later
    HOST = '0.0.0.0'  # Makes the server accessible externally
    PORT = 5000       # Standard port for Flask apps
    DEBUG_MODE = True # Set to False in production

    print(f"Starting Flask API server on {HOST}:{PORT} (Debug: {DEBUG_MODE})")
    # The following check is mostly for when __file__ might not be defined correctly
    if preprocess_data is None:
        print("Reminder: preprocess_data function was not loaded. Predictions may fail or be incorrect.")
    if model is None:
        print("Reminder: Model was not loaded. /predict endpoint will not work.")

    app.run(host=HOST, port=PORT, debug=DEBUG_MODE)

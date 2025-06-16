from pathlib import Path
import os
import logging

# Project Root
# Assuming config.py is in src/, so parent of parent is project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw_data'
PROCESSED_DATA_DIR = PROJECT_ROOT / 'data' / 'processed_data'
# Example filenames used in main blocks, can be configured here if needed
DEFAULT_RAW_FILENAME = 'canadian_gp_2023.csv'
DEFAULT_PROCESSED_FILENAME = 'processed_canadian_gp.csv'

# Model Paths
MODEL_DIR = PROJECT_ROOT / 'models'
DEFAULT_MODEL_FILENAME = 'random_forest_model.joblib'
MODEL_PATH = MODEL_DIR / DEFAULT_MODEL_FILENAME

# Logging Configuration
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_FILENAME = 'app.log'
LOG_FILE_PATH = LOG_DIR / LOG_FILENAME
LOG_LEVEL = logging.INFO

# Model Training Parameters
DEFAULT_MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': 42
}
# Example parameters for main block in modeling.py
EXAMPLE_TRAINING_PARAMS = {
    'n_estimators': 150,
    'random_state': 42,
    'max_depth': 10
}

# API Configuration
# Use environment variables for sensitive or environment-specific settings,
# with fallbacks to defaults from this config file.
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 5000))
API_DEBUG_MODE = os.getenv('API_DEBUG_MODE', 'True').lower() == 'true'

# FastF1 Cache (already defined in data_collection.py but good to centralize conceptually)
# FASTF1_CACHE_DIR = PROJECT_ROOT / '.fastf1_cache'
# Note: data_collection.py already handles its creation and usage.
# No need to redefine here unless we refactor data_collection.py to use this config for it.

if __name__ == '__main__':
    # Create directories if they don't exist when config is run directly (optional, for setup)
    # Usually, scripts using these paths should ensure they exist before writing.
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"PROCESSED_DATA_DIR: {PROCESSED_DATA_DIR}")
    print(f"MODEL_DIR: {MODEL_DIR}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print(f"LOG_DIR: {LOG_DIR}")
    print(f"LOG_FILE_PATH: {LOG_FILE_PATH}")
    print(f"API_HOST: {API_HOST}")
    print(f"API_PORT: {API_PORT}")
    print(f"API_DEBUG_MODE: {API_DEBUG_MODE}")

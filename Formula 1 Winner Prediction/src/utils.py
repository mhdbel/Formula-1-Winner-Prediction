import logging
from pathlib import Path
from src import config

def setup_logger(log_filename=None, log_dir=None, level=None):
    """
    Set up a logger for the application.

    Args:
        log_filename (str, optional): Name of the log file. Defaults to config.LOG_FILENAME.
        log_dir (str, optional): Directory to store log files. Defaults to config.LOG_DIR.
        level (int, optional): Logging level. Defaults to config.LOG_LEVEL.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_d = config.LOG_DIR if log_dir is None else Path(log_dir)
    log_fn = config.LOG_FILENAME if log_filename is None else log_filename
    log_level_to_use = config.LOG_LEVEL if level is None else level

    log_file_path = log_d / log_fn

    try:
        # Ensure the directory for the log file exists
        log_d.mkdir(parents=True, exist_ok=True)

        # Configure logging
        logging.basicConfig(
            filename=log_file_path,
            level=log_level_to_use,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        logger = logging.getLogger(__name__)
        print(f"Logger configured. Log file at: {log_file_path}")
        return logger

    except Exception as e:
        print(f"Error setting up logger at {log_file_path}: {e}. Falling back to console logging.")
        logging.basicConfig(
            level=log_level_to_use,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

# Initialize a default logger for the module
# This makes it easy to import `logger` from `utils` elsewhere.
# Uses the defaults from config.py by not passing arguments
logger = setup_logger()

if __name__ == '__main__':
    # Example usage of the logger
    logger.info("This is an informational message from utils.py using config-based logger.")
    logger.warning("This is a warning message from utils.py.")
    logger.error("This is an error message from utils.py.")

    # Example of using a logger with a different configuration (overriding defaults)
    # test_logger = setup_logger(log_filename='test.log', level=logging.DEBUG)
    # test_logger.debug("This is a debug message for the test logger.")
    print(f"Check the '{config.LOG_DIR / config.LOG_FILENAME}' (and '{config.LOG_DIR / 'test.log'}' if uncommented) for output.")

import logging
from pathlib import Path

def setup_logger(log_filename='app.log', log_dir='logs', level=logging.INFO):
    """
    Set up a logger for the application.

    Args:
        log_filename (str): Name of the log file.
        log_dir (str): Directory to store log files.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_directory = Path(log_dir)
    log_file_path = log_directory / log_filename

    try:
        # Ensure the directory for the log file exists
        log_directory.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        # Using a rotating file handler is often a good idea for long-running apps
        # For simplicity here, we'll stick to basicConfig, but consider RotatingFileHandler
        logging.basicConfig(
            filename=log_file_path,
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Get the root logger if no specific name is used, or get a named logger
        logger = logging.getLogger(__name__) # Using __name__ for module-specific logger
        if not logger.handlers: # Ensure handlers are not added multiple times if setup_logger is called again
             # This basicConfig sets up the root logger. If you want a specific named logger
             # that doesn't propagate to root or has its own handlers, you'd configure it differently.
             # For most simple applications, basicConfig is fine.
             pass # basicConfig configures the root logger.

        print(f"Logger configured. Log file at: {log_file_path}")
        return logger

    except Exception as e:
        # Fallback to basic console logging if file logging setup fails
        print(f"Error setting up logger at {log_file_path}: {e}. Falling back to console logging.")
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)

# Initialize a default logger for the module
# This makes it easy to import `logger` from `utils` elsewhere.
logger = setup_logger()

if __name__ == '__main__':
    # Example usage of the logger
    logger.info("This is an informational message from utils.py.")
    logger.warning("This is a warning message from utils.py.")
    logger.error("This is an error message from utils.py.")
    
    # Example of using a logger with a different configuration
    # test_logger = setup_logger(log_filename='test.log', level=logging.DEBUG)
    # test_logger.debug("This is a debug message for the test logger.")
    print("Check the 'logs/app.log' (and 'logs/test.log' if uncommented) for output.")

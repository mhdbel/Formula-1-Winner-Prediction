import logging

def setup_logger():
    """
    Set up a logger for the application.
    """
    logging.basicConfig(filename='logs/app.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logger()
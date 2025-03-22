import logging
import datetime
import os

def setup_logging():
    """Configure logging for the Weinstein Analyzer application"""
    log_filename = f"weinstein_analyzer_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger('WeinsteinAnalyzer')
    
    return logger

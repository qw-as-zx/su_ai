import logging
import sys
from pathlib import Path
from datetime import datetime

from app.utils.config import settings

# Global logger dictionary to keep track of all created loggers
LOGGERS = {}

def get_logger(name="app"):
    """Get or create a logger with the given name."""
    global LOGGERS
    
    if name in LOGGERS:
        return LOGGERS[name]
        
    logger = logging.getLogger(name)
    LOGGERS[name] = logger
    return logger

def configure_logging():
    """Configure structured logging for the application"""
    LOG_DIR = Path("logs")
    LOG_DIR.mkdir(exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = LOG_DIR / f"app_{current_time}.log"
    
    # Check if logging has already been configured
    if logging.getLogger().handlers:
        return get_logger("app")
    
    # Configure the root logger
    logging.basicConfig(
        level=logging.DEBUG,
        # level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
 
    # Configure third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("redis").setLevel(logging.INFO)
    logging.getLogger("langchain").setLevel(logging.INFO)
    
    # Create app logger
    app_logger = get_logger("app")
    app_logger.info("Logging configured successfully")
    return app_logger

# Initialize the root logger - will be called when this module is imported
root_logger = configure_logging()
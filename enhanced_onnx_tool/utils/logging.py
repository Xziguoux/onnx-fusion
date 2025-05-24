"""
Logging utilities for Enhanced ONNX Tool.
"""

import logging
import sys
from typing import Optional

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Global logger dictionary
_loggers = {}

def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    # Add handler to root logger
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    global _loggers
    
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    
    return logger

def set_log_level(level: int) -> None:
    """
    Set log level for all loggers.
    
    Args:
        level: Logging level
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Update existing loggers
    for logger in _loggers.values():
        logger.setLevel(level)

def add_file_handler(filename: str, level: Optional[int] = None) -> None:
    """
    Add a file handler to the root logger.
    
    Args:
        filename: Log file path
        level: Logging level for file handler
    """
    root_logger = logging.getLogger()
    
    # Create file handler
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    
    if level is not None:
        file_handler.setLevel(level)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)

# Initialize logging
setup_logging()

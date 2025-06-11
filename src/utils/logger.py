"""
Logging utilities for the Agentic RAG Workflow Engine.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import datetime

# Global logger configuration
_loggers = {}
_log_level = logging.INFO
_log_file = None

def setup_logger(
    name: str, 
    level: int = logging.INFO, 
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    global _log_level, _log_file
    _log_level = level
    _log_file = log_file
    
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    _loggers[name] = logger
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the current global configuration.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    return setup_logger(name, _log_level, _log_file)

# Initialize default logger
default_logger = setup_logger("agentic_rag", logging.INFO)
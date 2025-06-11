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

def set_log_level(level: int):
    """Set the global log level for all loggers."""
    global _log_level
    _log_level = level
    
    for logger in _loggers.values():
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

def enable_file_logging(log_file: str):
    """Enable file logging for all existing loggers."""
    global _log_file
    _log_file = log_file
    
    # Add file handler to existing loggers
    for logger in _loggers.values():
        # Check if file handler already exists
        has_file_handler = any(
            isinstance(handler, logging.FileHandler) 
            for handler in logger.handlers
        )
        
        if not has_file_handler:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(_log_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

class WorkflowLogger:
    """Specialized logger for workflow tracking and observability."""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.workflow_id = None
        self.step_count = 0
    
    def start_workflow(self, query: str) -> str:
        """Start a new workflow and return workflow ID."""
        self.workflow_id = f"wf_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.step_count = 0
        
        self.logger.info(f"[{self.workflow_id}] Starting workflow for query: {query}")
        return self.workflow_id
    
    def log_step(self, agent: str, action: str, details: str = None, duration: float = None):
        """Log a workflow step."""
        self.step_count += 1
        
        log_message = f"[{self.workflow_id}] Step {self.step_count} - {agent}: {action}"
        
        if duration:
            log_message += f" (took {duration:.3f}s)"
        
        if details:
            log_message += f" - {details}"
        
        self.logger.info(log_message)
    
    def log_error(self, agent: str, error: str, step: str = None):
        """Log a workflow error."""
        error_message = f"[{self.workflow_id}] ERROR in {agent}"
        if step:
            error_message += f" during {step}"
        error_message += f": {error}"
        
        self.logger.error(error_message)
    
    def end_workflow(self, success: bool, total_duration: float = None):
        """End the workflow logging."""
        status = "COMPLETED" if success else "FAILED"
        
        log_message = f"[{self.workflow_id}] Workflow {status}"
        if total_duration:
            log_message += f" in {total_duration:.3f}s"
        
        if success:
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)

# Performance monitoring utilities
class PerformanceMonitor:
    """Monitor and log performance metrics."""
    
    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        import time
        self.metrics[operation] = {"start_time": time.time()}
    
    def end_timer(self, operation: str, log_result: bool = True):
        """End timing an operation and optionally log the result."""
        import time
        
        if operation not in self.metrics:
            self.logger.warning(f"Timer for operation '{operation}' was not started")
            return None
        
        end_time = time.time()
        duration = end_time - self.metrics[operation]["start_time"]
        self.metrics[operation]["duration"] = duration
        self.metrics[operation]["end_time"] = end_time
        
        if log_result:
            self.logger.info(f"Operation '{operation}' completed in {duration:.3f}s")
        
        return duration
    
    def get_metrics(self) -> dict:
        """Get all recorded metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()

# Initialize default logger
default_logger = setup_logger("agentic_rag", logging.INFO)
